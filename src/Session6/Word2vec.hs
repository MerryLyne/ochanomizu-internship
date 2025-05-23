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
import Torch.Serialize (loadParams, saveParams)
import Torch.Tensor (Tensor, asTensor, shape, asValue)
import Torch.TensorFactories (randnIO, eye', zeros')
import Torch.Optim (Adam(..), runStep, mkAdam)
import Control.Monad (foldM)
import Torch.DType (DType(..))
import Torch.TensorOptions (withDType, defaultOpts)
import Torch.Device (Device(..), DeviceType(..))

textFilePath = "sample.txt"
modelPath = "sample_embedding.params"
wordLstPath = "sample_wordlst.txt"

data EmbeddingSpec = EmbeddingSpec
  { wordNum :: Int
  , wordDim :: Int
  }
  deriving (Show, Eq, Generic)

data Embedding = Embedding
  { wordEmbedding :: Parameter
  }
  deriving (Show, Generic, Parameterized)

isWord ::
  BL.ByteString ->
  [BL.ByteString]
isWord bs =
  let raw = BL.words $ BL.map normalizeChar bs
      normalizeChar c
        | isAlpha c = toLower c
        | otherwise = ' '
  in filter (not . BL.null) raw

preprocess ::
  B.ByteString ->
  [[B.ByteString]]
preprocess texts =
  map isWord $ BL.lines texts

toyEmbedding ::
  EmbeddingSpec ->
  Tensor
toyEmbedding EmbeddingSpec{..} =
  eye' wordNum wordDim

wordToIndexFactory ::
  [B.ByteString] ->
  (B.ByteString -> Int)
wordToIndexFactory wordlst =
  let wordMap = M.fromList (zip wordlst [0..])
      unkIdx = length wordlst - 1
  in \wrd -> M.findWithDefault unkIdx wrd wordMap

generateSGPairs :: 
  Int -> 
  [Int] -> 
  [(Int, Int)]
generateSGPairs windowSize tokens
  | length tokens < 2 = [] 
  | otherwise = concatMap (extractPairs tokens) [0 .. length tokens - 1]
  where
    extractPairs :: [Int] -> Int -> [(Int, Int)]
    extractPairs xs i
      | i < 0 || i >= length xs = []  
      | otherwise =
          let center = xs !! i
              start = max 0 (i - windowSize)
              end = min (length xs) (i + windowSize + 1)
              context = [xs !! j | j <- [start .. end - 1], j /= i, j >= 0, j < length xs]
          in [(center, c) | c <- context]

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
  let
    inputVecs = embedding' (toDependent $ wordEmbedding inputEmbedding) targetWordIdxs
    outputMatT = transpose (Dim 0) (Dim 1) (toDependent $ wordEmbedding outputEmbedding)
    scores = inputVecs `matmul` outputMatT
  in
    scores

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
  print ("Word list :", wordlst)
  let indexedSentences = map (map wordToIndex) wordLines
  putStrLn $ "Indexed sentences sample (first 3): " ++ show (take 3 indexedSentences)
  putStrLn $ "Max index in indexedSentences: " ++ show (foldr (max . maximum) 0 indexedSentences)
  putStrLn $ "Exemple de phrase indexée : " ++ show (take 5 $ head indexedSentences)
  let embsddingSpec = EmbeddingSpec {wordNum = length wordlst, wordDim = 9}
  wordEmb <- makeIndependent $ toyEmbedding embsddingSpec
  let emb = Embedding { wordEmbedding = wordEmb }
  let windowSize = 2
      sgPairs = concatMap (generateSGPairs windowSize) indexedSentences
  putStrLn $ "Nb pairs : " ++ show (length sgPairs)
  putStrLn $ "Pairs : " ++ show (take 5 sgPairs)
  let vocabSize = length wordlst
      embeddingDim = 9
      skipGramSpec = SkipGramSpec { vocabSize = vocabSize, embeddingDim = embeddingDim }
  initialModel <- sample skipGramSpec -- This line's indentation was causing the issue
  putStrLn "Forme du tenseur d'embedding d'entrée initial :"
  print $ Torch.Tensor.shape (toDependent $ wordEmbedding $ inputEmbedding initialModel)
  let sampleTxt = B.pack $ encode "This is awesome.\nmodel is developing"
      idxes = map (map wordToIndex) (preprocess sampleTxt)
  print ("Index:", idxes)

  let learningRate = 0.005 :: Float
  let optimizer = Torch.Optim.mkAdam 0 0.9 0.999 (flattenParameters initialModel)

  let numEpochs = 50
  putStrLn $ "Starting training for " ++ show numEpochs ++ " epochs..."

  (trainedModel, finalOptimizer, _, _) <- foldM
    (\(model, opt, accLoss, count) epoch -> do
      putStrLn $ "\nEpoch " ++ show epoch
      (currentModel, currentOpt, totalLoss, numPairsProcessed) <-
        foldM
          (\(m, o, currentAccLoss, currentCount) (centerIdx, contextIdx) -> do
            let centerTensor = asTensor (fromIntegral centerIdx :: Int)
            let scores = forward m centerTensor
            let loss = crossEntropyLoss scores contextIdx
            (newParams, newOpt) <- runStep (flattenParameters m) o (asTensor learningRate) loss
            let newModel = replaceParameters m newParams
            let newAccLoss = currentAccLoss + (asValue loss :: Float)
            return (newModel, newOpt, newAccLoss, currentCount + 1)
          )
          (model, opt, 0.0 :: Float, 0)
          sgPairs
      let avgLoss = if numPairsProcessed > 0
                    then totalLoss / fromIntegral numPairsProcessed
                    else 0.0 :: Float
      putStrLn $ "   Average loss for epoch: " ++ show avgLoss
      return (currentModel, currentOpt, 0.0 :: Float, 0)
    ) (initialModel, optimizer, 0.0 :: Float, 0) [1 .. numEpochs]
  putStrLn "Training complete here"
  saveParams (inputEmbedding trainedModel) modelPath
  B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)
  putStrLn $ "Word list saved to: " ++ wordLstPath
  putStrLn $ "Shape of the trained input embedding: "
  print $ Torch.Tensor.shape (toDependent $ wordEmbedding $ inputEmbedding trainedModel)
  return ()