{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

module Session5.Data (
    readAdmissions,
    loadTrainData,
    loadTestData,
    loadEvalData,
    featuresToTensor,
    labelsToTensor,
    cgpaToTensor,
    prepareData
) where

import Torch.Tensor (Tensor, asTensor)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import Data.Csv
import Control.Monad (mzero)
import GHC.Generics (Generic)

data Admission = Admission
  { serialNo         :: !Int
  , gre              :: !Int
  , toefl            :: !Int
  , uniRating        :: !Int
  , sop              :: !Double
  , lor              :: !Double
  , cgpa             :: !Double
  , research         :: !Int
  , chanceOfAdmit    :: !Double
  } deriving (Show, Generic)

instance FromNamedRecord Admission where
  parseNamedRecord m = Admission
    <$> m .: "Serial No."
    <*> m .: "GRE Score"
    <*> m .: "TOEFL Score"
    <*> m .: "University Rating"
    <*> m .: "SOP"
    <*> m .: "LOR "
    <*> m .: "CGPA"
    <*> m .: "Research"
    <*> m .: "Chance of Admit "

readAdmissions :: FilePath -> IO (Either String (V.Vector Admission))
readAdmissions path = do
  raw <- BL.readFile path
  case decodeByName  raw of
    Left err -> return $ Left err
    Right (_, vec) -> return $ Right vec

loadTrainData :: IO (Either String (V.Vector Admission))
loadTrainData = readAdmissions "train_data.csv"

loadTestData :: IO (Either String (V.Vector Admission))
loadTestData = readAdmissions "test_data.csv"

loadEvalData :: IO (Either String (V.Vector Admission))
loadEvalData = readAdmissions "eval.csv"

featuresToTensor :: V.Vector Admission -> Tensor
featuresToTensor = asTensor . V.toList . V.map toFeatureList
  where
    toFeatureList adm =
      [ fromIntegral (gre adm) / 340.0
      , fromIntegral (toefl adm) / 120.0
      , fromIntegral (uniRating adm) / 5.0
      , realToFrac (sop adm) / 5.0
      , realToFrac (lor adm) / 5.0
      , realToFrac (cgpa adm) / 10.0
      , fromIntegral (research adm) :: Float
      ]

labelsToTensor :: V.Vector Admission -> Tensor
labelsToTensor = asTensor . V.toList . V.map (realToFrac . chanceOfAdmit :: Admission -> Float)

cgpaToTensor :: V.Vector Admission -> Tensor
cgpaToTensor = asTensor . V.toList . V.map (realToFrac . cgpa :: Admission -> Float)

prepareData :: V.Vector Admission -> (Tensor, Tensor)
prepareData admissions = (featuresToTensor admissions, labelsToTensor admissions)