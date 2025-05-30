{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}

module Session7.RNN where
  
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics
import Torch.NN (Parameter, Parameterized(..), Randomizable(..))
import Torch.Serialize (loadParams)
import Torch.TensorFactories (randnIO')
import Torch.Autograd (makeIndependent)
import Torch.Tensor (Tensor)
import Torch.Functional (add, matmul, tanh, transpose)
import qualified Torch.Functional as F
import Torch.Autograd (toDependent)
import qualified Torch.Serialize as Serialize


linear :: Tensor -> Tensor -> Tensor -> Tensor
linear weight bias input = add (matmul input (transpose2D weight)) bias

transpose2D :: Tensor -> Tensor
transpose2D = F.transpose2D

-- amazon review data
data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic)

instance FromJSON Image
instance ToJSON Image

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
  } deriving (Show, Generic)

instance FromJSON AmazonReview
instance ToJSON AmazonReview

-- model
data ModelSpec = ModelSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)


data Model = Model {
  emb :: Embedding,
  rnn :: RNNCell
} deriving (Show, Generic, Parameterized)

rnnCell :: RNNCell -> Tensor -> Tensor -> Tensor
rnnCell RNNCell{..} input hidden =
  let w_ih' = toDependent w_ih
      w_hh' = toDependent w_hh
      b_h'  = toDependent b_h
  in F.tanh $ F.add (F.add (F.matmul input (transpose2D w_ih')) b_h') (F.matmul hidden (transpose2D w_hh'))


data RNNCell = RNNCell
  { w_ih :: Parameter  -- input weights
  , w_hh :: Parameter  -- hidden weights
  , b_h  :: Parameter  -- biais
  } deriving (Show, Generic, Parameterized)

data RNNCellSpec = RNNCellSpec
  { inputSize :: Int
  , hiddenSize :: Int
  } deriving (Show, Eq)

instance Randomizable RNNCellSpec RNNCell where
  sample RNNCellSpec{..} = do
    w_ih <- makeIndependent =<< randnIO' [hiddenSize, inputSize]
    w_hh <- makeIndependent =<< randnIO' [hiddenSize, hiddenSize]
    b_h  <- makeIndependent =<< randnIO' [hiddenSize]
    return $ RNNCell w_ih w_hh b_h

instance Randomizable ModelSpec Model where
  sample ModelSpec{..} = do
    wordEmbedding <- makeIndependent =<< randnIO' [wordNum, wordDim]
    let emb = Embedding wordEmbedding
    rnn <- sample (RNNCellSpec wordDim wordDim)
    return $ Model emb rnn


-- randomize and initialize embedding with loaded params
--initialize ::
--  ModelSpec ->
--  FilePath ->
--  IO Model
--initialize modelSpec embPath = do
--  randomizedModel <- sample modelSpec
--  loadedEmb <- loadParams (emb randomizedModel) embPath
--  return randomizedModel { emb = loadedEmb }

initialize :: ModelSpec -> IO Model
initialize modelSpec = sample modelSpec


-- your amazon review json
amazonReviewPath :: FilePath
amazonReviewPath = "train.jsonl"

outputPath :: FilePath
outputPath = "review-texts.txt"

embeddingPath =  "sample_embedding.params"

wordLstPath = "sample_wordlst.txt"

decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview] 
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
  in sequenceA $ map eitherDecode jsonList

saveModelParams :: Parameterized model => model -> FilePath -> IO ()
saveModelParams model path = Serialize.saveParams model path

main :: IO ()
main = do
  jsonl <- B.readFile amazonReviewPath
  let amazonReviews = decodeToAmazonReview jsonl
  let reviews = case amazonReviews of
                  Left err -> []
                  Right reviews -> reviews

  -- load word list (It's important to use the same list as whan creating embeddings)
  wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)

  -- load params (set　wordDim　and wordNum same as session5)
  let modelSpec = ModelSpec {
    wordDim = 9, 
    wordNum = 9
  }
--  initModel <- initialize modelSpec embeddingPath
  initModel <- initialize modelSpec
  print initModel
  return ()