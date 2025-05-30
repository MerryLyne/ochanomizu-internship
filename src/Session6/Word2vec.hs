{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Session6.Word2vec where

import Codec.Binary.UTF8.String (encode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Char (isAlpha, toLower)
import Data.List (nub)
import qualified Data.Map.Strict as M
import Data.Word (Word8)
import GHC.Generics
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding', matmul, transpose, Dim(..), logSoftmax, nllLoss')
import Torch.NN (flattenParameters, Parameter, Parameterized (..), Randomizable(..), replaceParameters)
import Torch.Serialize (saveParams)
import Torch.Tensor (Tensor, asTensor, shape, asValue)
import Torch.TensorFactories (randnIO, eye')
import Torch.Optim (runStep, mkAdam)
import Torch.DType (DType(..))
import Torch.TensorOptions (withDType, defaultOpts)
import Control.Monad (foldM)

textFilePath = "sample.txt"
modelPath = "sample_embedding.params"
wordLstPath = "sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec
  { wordNum :: Int
  , wordDim :: Int
  } deriving (Show, Eq, Generic)

data Embedding = Embedding
  { wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

isWord :: BL.ByteString -> [BL.ByteString]
isWord bs =
  let normalize c = if isAlpha c then toLower c else ' '
  in filter (not . BL.null) $ BL.words $ BL.map normalize bs

preprocess :: B.ByteString -> [[B.ByteString]]
preprocess = map isWord . BL.lines

toyEmbedding :: EmbeddingSpec -> Tensor
toyEmbedding EmbeddingSpec{..} = eye' wordNum wordDim

wordToIndexFactory :: [B.ByteString] -> B.ByteString -> Int
wordToIndexFactory wordlst =
  let m = M.fromList $ zip wordlst [1..]
  in \w -> M.findWithDefault 0 w m

generateSGPairs ::
  Int ->
  Int ->
  [Int] ->
  [(Int, Int)]
generateSGPairs vocabSize windowSize tokens
  | length tokens < 2 = []
  | otherwise = concatMap (extractPairs tokens) [0 .. length tokens - 1]
  where
    extractPairs xs i
      | i < 0 || i >= length xs = []
      | otherwise =
          case safeIndex xs i of
            Nothing -> []
            Just center ->
              let start = max 0 (i - windowSize)
                  end = min (length xs) (i + windowSize + 1)
                  context = [j | j <- [start .. end - 1], j /= i, j >= 0, j < length xs]
                  pairs = if center < vocabSize
                            then [(center, c) | j <- context, Just c <- [safeIndex xs j], c >= 0, c < vocabSize]
                            else []
              in pairs

    safeIndex :: [a] -> Int -> Maybe a
    safeIndex xs idx = if idx < 0 || idx >= length xs then Nothing else Just (xs !! idx)

data SkipGramSpec = SkipGramSpec
  { vocabSize :: Int
  , embeddingDim :: Int
  } deriving (Show, Eq, Generic)

data SkipGram = SkipGram
  { inputEmbedding :: Embedding
  , outputEmbedding :: Embedding
  } deriving (Generic, Show, Parameterized)

instance Randomizable SkipGramSpec SkipGram where
  sample SkipGramSpec{..} = do
    let opts = withDType Float defaultOpts
    inp <- makeIndependent =<< randnIO [vocabSize, embeddingDim] opts
    out <- makeIndependent =<< randnIO [vocabSize, embeddingDim] opts
    return $ SkipGram
      { inputEmbedding = Embedding inp
      , outputEmbedding = Embedding out
      }

forward :: SkipGram -> Tensor -> Tensor
forward SkipGram{..} targetWordIdxs =
  let inputVecs = embedding' (toDependent $ wordEmbedding inputEmbedding) targetWordIdxs
      outputMatT = transpose (Dim 0) (Dim 1) (toDependent $ wordEmbedding outputEmbedding)
  in inputVecs `matmul` outputMatT

crossEntropyLoss :: Tensor -> Int -> Tensor
crossEntropyLoss scores targetContextIdx =
  let logProbs = logSoftmax (Dim 0) scores
      targetTensor = asTensor (fromIntegral targetContextIdx :: Int)
  in nllLoss' targetTensor logProbs

main :: IO ()
main = do
  texts <- B.readFile textFilePath
  let wordLines = preprocess texts
      wordlst = nub $ concat wordLines
      wordToIndex = wordToIndexFactory wordlst
      indexedSentences = map (map wordToIndex) wordLines

  putStrLn ("Word list size: " ++ show (length wordlst))
  let vocabSize = length wordlst + 1
  putStrLn ("Vocab size (incl. <unk>): " ++ show vocabSize)
  putStrLn ("Sample indexed sentences: " ++ show (take 3 indexedSentences))
  putStrLn ("Max index in data: " ++ show (maximum (concat indexedSentences)))

  let embSpec = EmbeddingSpec { wordNum = vocabSize, wordDim = 9 }
  wordEmb <- makeIndependent $ toyEmbedding embSpec
  let emb = Embedding { wordEmbedding = wordEmb }

  let windowSize = 2
      sgPairs = concatMap (generateSGPairs vocabSize windowSize) indexedSentences

  putStrLn ("Nb pairs : " ++ show (length sgPairs))
  putStrLn ("Pairs : " ++ show (take 5 sgPairs))

  let embeddingDim = 9
      skipGramSpec = SkipGramSpec { vocabSize = vocabSize, embeddingDim = embeddingDim }
  initialModel <- sample skipGramSpec

  putStrLn "Forme du tenseur d'embedding d'entrée initial :"
  print $ shape $ toDependent $ wordEmbedding $ inputEmbedding initialModel

  let learningRate = 0.005 :: Float
      optimizer = mkAdam 0 0.9 0.999 (flattenParameters initialModel)
      numEpochs = 50

  putStrLn $ "Starting training for " ++ show numEpochs ++ " epochs..."

  (trainedModel, _, _, _) <- foldM
    (\(model, opt, accLoss, count) epoch -> do
      putStrLn $ "\nEpoch " ++ show epoch
      (currentModel, currentOpt, totalLoss, totalCount) <- foldM
        (\(m, o, l, c) (centerIdx, contextIdx) -> do
          let centerTensor = asTensor (fromIntegral centerIdx :: Int)
              scores = forward m centerTensor
              loss = crossEntropyLoss scores contextIdx
          (newParams, newOpt) <- runStep (flattenParameters m) o (asTensor learningRate) loss
          let newModel = replaceParameters m newParams
              newLoss = l + asValue loss
          return (newModel, newOpt, newLoss, c + 1))
        (model, opt, 0.0 :: Float, 0)
        sgPairs
      let avgLoss = if totalCount > 0 then totalLoss / fromIntegral totalCount else 0.0
      putStrLn $ "Average loss for epoch: " ++ show avgLoss
      return (currentModel, currentOpt, 0.0, 0))
    (initialModel, optimizer, 0.0, 0)
    [1..numEpochs]

  putStrLn "Training complete."
  saveParams (inputEmbedding trainedModel) modelPath
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  putStrLn "Model and word list saved."
