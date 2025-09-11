module Session5.Admit where

import Prelude hiding (tanh)
import Torch
import Torch.Device (Device (..), DeviceType (..))
import Torch.Functional (unsqueeze)
import Torch.Layer.MLP (ActName (..), MLPHypParams (..), mlpLayer)
import Torch.NN (flattenParameters, sample)
import Torch.Optim (mkAdam)
import Torch.Tensor (asTensor, asValue, shape)
import Torch.Train (showLoss, update)
import qualified Torch.Functional as F
import Torch.Control (mapAccumM)
import qualified Data.Vector as V

import ML.Exp.Chart (drawLearningCurve)
import Session5.Data (prepareData, readAdmissions)
import Session5.Evaluation (accuracy, f1Score, precision, recall)

main :: IO ()
main = do
  let iter = 1000 :: Int
      device = Device CPU 0
      hypParams = MLPHypParams device 7 [(16, Relu), (16, Relu), (1, Sigmoid)]

  trainData <- readAdmissions "train_data.csv"
  testData <- readAdmissions "test_data.csv"
  evalData <- readAdmissions "eval.csv"

  case (trainData, testData, evalData) of
    (Right trainVec, Right testVec, Right evalVec) -> do
      putStrLn $ "Number of training admissions: " ++ show (V.length trainVec)
      let (trainX, trainYRaw) = prepareData trainVec
          trainY = unsqueeze (Dim 1) trainYRaw
      putStrLn $ "Shape of trainX: " ++ show (shape trainX)
      putStrLn $ "Shape of trainY: " ++ show (shape trainY)

      initModel <- sample hypParams
      let initOpt = mkAdam 0 0.9 0.999 (flattenParameters initModel)

      ((trainedModel, _), losses) <- mapAccumM [1 .. iter] (initModel, initOpt) $ \epoch (model, opt) -> do
        let predY = mlpLayer model trainX
            loss = F.mseLoss trainY predY
            lossValue = asValue loss :: Float
        showLoss 100 epoch lossValue
        (updatedModel, updatedOpt) <- update model opt loss 1e-4
        return ((updatedModel, updatedOpt), lossValue)

      drawLearningCurve "graph.png" "Training Loss Curve" [("Loss", reverse losses)]

      let sampleInput = asTensor [330 / 340, 115 / 120, 4 / 5, 4.5 / 5, 4.5 / 5, 9.2 / 10, 1 :: Float]
      print (mlpLayer trainedModel sampleInput)

      putStrLn $ "Number of testing admissions: " ++ show (V.length testVec)
      let (testX, testYRaw) = prepareData testVec
          testY = unsqueeze (Dim 1) testYRaw
      putStrLn $ "Shape of testX: " ++ show (shape testX)
      putStrLn $ "Shape of testY: " ++ show (shape testY)

      let predTensor = mlpLayer trainedModel testX
      putStrLn "\nRaw Predictions Before Thresholding (first 10):"
      print (Prelude.take 10 (asValue predTensor :: [[Float]]))

      let predLabelsTensorBool = F.gt predTensor (asTensor (0.5 :: Float))
          predLabelsTensor = F.toDType Float predLabelsTensorBool
          trueLabelsTensor = testY

      putStrLn "\nRaw Predictions on Test Data (first 10):"
      print (Prelude.take 10 (asValue predTensor :: [[Float]]))

      putStrLn "\nThresholded Predictions on Test Data (first 10):"
      print (Prelude.take 10 (asValue predLabelsTensor :: [[Float]]))

      putStrLn "\nTrue Labels on Test Data (first 10):"
      print (Prelude.take 10 (asValue trueLabelsTensor :: [[Float]]))

      putStrLn "\nEvaluation :"
      putStrLn $ "Accuracy:  " ++ show (accuracy trueLabelsTensor predLabelsTensor)
      putStrLn $ "Precision: " ++ show (precision trueLabelsTensor predLabelsTensor)
      putStrLn $ "Recall:    " ++ show (recall trueLabelsTensor predLabelsTensor)
      putStrLn $ "F1 Score:  " ++ show (f1Score trueLabelsTensor predLabelsTensor)