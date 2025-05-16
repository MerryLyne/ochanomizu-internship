module Session5.Evaluation where

import Torch
import Torch.Functional (eq, mul, sumAll)
import Data.Int (Int64)

truePositives :: Tensor -> Tensor -> Tensor
truePositives actualOutput predictedOutput = 
    let tp = mul (eq actualOutput (asTensor (1 :: Int64)))
                 (eq predictedOutput (asTensor (1 :: Int64)))
    in sumAll tp

trueNegatives :: Tensor -> Tensor -> Tensor
trueNegatives actualOutput predictedOutput =
    let tn = mul (eq actualOutput (asTensor (0 :: Int64)))
                 (eq predictedOutput (asTensor (0 :: Int64)))
    in sumAll tn

falsePositives :: Tensor -> Tensor -> Tensor
falsePositives actualOutput predictedOutput =
    let fp = mul (eq actualOutput (asTensor (0 :: Int64)))
                 (eq predictedOutput (asTensor (1 :: Int64)))
    in sumAll fp

falseNegatives :: Tensor -> Tensor -> Tensor
falseNegatives actualOutput predictedOutput =
    let fn = mul (eq actualOutput (asTensor (1 :: Int64)))
                 (eq predictedOutput (asTensor (0 :: Int64)))
    in sumAll fn

accuracy :: Tensor -> Tensor -> Float
accuracy actualOutput predictedOutput =
    let tp = asValue (truePositives actualOutput predictedOutput) :: Int64
        tn = asValue (trueNegatives actualOutput predictedOutput) :: Int64
        fp = asValue (falsePositives actualOutput predictedOutput) :: Int64
        fn = asValue (falseNegatives actualOutput predictedOutput) :: Int64
        total = tp + tn + fp + fn
    in if total == 0
       then 0.0
       else fromIntegral (tp + tn) / fromIntegral total

precision :: Tensor -> Tensor -> Float
precision actualOutput predictedOutput =
    let tp = asValue $ truePositives actualOutput predictedOutput :: Int64
        fp = asValue $ falsePositives actualOutput predictedOutput :: Int64
    in if (tp + fp) == 0
       then 0.0
       else fromIntegral tp / fromIntegral (tp + fp)

recall :: Tensor -> Tensor -> Float
recall actualOutput predictedOutput =
    let tp = asValue $ truePositives actualOutput predictedOutput :: Int64
        fn = asValue $ falseNegatives actualOutput predictedOutput :: Int64
    in if (tp + fn) == 0
       then 0.0
       else fromIntegral tp / fromIntegral (tp + fn)

confusionMatrix :: [Int] -> [Int] -> (Int, Int, Int, Int)
confusionMatrix preds labels =
  let paired = zip preds labels
      tp = length [() | (p, l) <- paired, p == 1 && l == 1]
      fp = length [() | (p, l) <- paired, p == 1 && l == 0]
      fn = length [() | (p, l) <- paired, p == 0 && l == 1]
      tn = length [() | (p, l) <- paired, p == 0 && l == 0]
  in (tp, fp, fn, tn)

f1Score :: Tensor -> Tensor -> Float
f1Score actualOutput predictedOutput =
    let prec = precision actualOutput predictedOutput
        rec = recall actualOutput predictedOutput
    in if (prec + rec) == 0
       then 0.0
       else 2 * (prec * rec) / (prec + rec)

main :: IO ()
main = do
    let actualOutput = asTensor ([1, 0, 1, 1, 0, 1, 0, 0] :: [Int64])
        predictedOutput = asTensor ([0, 0, 1, 0, 0, 1, 1, 0] :: [Int64])

    putStrLn $ "Accuracy: " ++ show (accuracy actualOutput predictedOutput)
    putStrLn $ "Precision: " ++ show (precision actualOutput predictedOutput)
    putStrLn $ "Recall: " ++ show (recall actualOutput predictedOutput)
    putStrLn $ "F1 Score: " ++ show (f1Score actualOutput predictedOutput)
