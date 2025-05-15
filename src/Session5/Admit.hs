{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Session5.Admit where

import Prelude hiding (tanh)
import Control.Monad (forM_)
import Torch
import Torch.Device (Device(..), DeviceType(..))
import Torch.NN (sample)
import Torch.Optim (mkAdam)
import Torch.Train (update, showLoss)
import Torch.Functional (mseLoss, unsqueeze)
import Torch.Tensor (asValue, asTensor, Tensor)
import Torch.Control (mapAccumM)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer)
import Data.Typeable (Typeable, typeOf)
import ML.Exp.Chart (drawLearningCurve)
import Session5.Data (readAdmissions, prepareData)

main :: IO ()
main = do
  let iter = 30 :: Int
      device = Device CPU 0
      hypParams = MLPHypParams device 7 [(16, Relu), (16, Relu), (1, Sigmoid)]
  
  trainData <- readAdmissions "train_data.csv"
  testData  <- readAdmissions "test_data.csv"
  evalData  <- readAdmissions "eval.csv"
  
  case (trainData, testData, evalData) of
    (Right trainVec, Right _, Right _) -> do
      let (trainX, trainYRaw) = prepareData trainVec
          trainY = unsqueeze (Dim 1) trainYRaw
      
      initModel <- sample hypParams
      let flattenedParams = Torch.flattenParameters initModel  -- Use fully qualified name
      print (typeOf flattenedParams)
      let initOpt = mkAdam 0 0.9 0.999 (Torch.flattenParameters initModel)  -- Use fully qualified name
      
      ((trainedModel, _), losses) <- mapAccumM [1..iter] (initModel, initOpt) $ \epoch (model, opt) -> do
        let predY = mlpLayer model trainX
            loss = mseLoss predY trainY
            lossValue = asValue loss :: Float
        showLoss 10 epoch lossValue
        (updatedModel, updatedOpt) <- update model opt loss 1e-4
        return ((updatedModel, updatedOpt), lossValue)
      
      drawLearningCurve "graph.png" "Training Loss Curve" [("Loss", losses)]
      
      let sampleInput = asTensor [330/340, 115/120, 4/5, 4.5/5, 4.5/5, 9.2/10, 1 :: Float]
      putStrLn "\nSample Prediction:"
      print (mlpLayer trainedModel sampleInput)
    
    _ -> putStrLn "Erreur de chargement des données (train, test, eval)."