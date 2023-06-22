import DatasetPreparation
import MlpConstruction
import GBRconstruction
import DatasetAnalisys
import BaseModels


if __name__ == '__main__':
    # DatasetPreparation.kFoldSplit(10)
    # MlpConstruction.kFoldCrossValidation()
    # MlpConstruction.adaptiveBoostingMLP()
    # MlpConstruction.evaluateMLP()
    MlpConstruction.evaluateBoostedMLP()
    # GBRconstruction.gridSearchGradientRegressor()
    # GBRconstruction.kFoldGradientBoostingRegressor()
    # GBRconstruction.gbrMeanResults()
    # MlpConstruction.testKModels()
    # BaseModels.kFoldCrossValidation("KNN")
    # BaseModels.kFoldCrossValidation("LR")
    # BaseModels.kFoldCrossValidation("SVM")
    # BaseModels.kFoldCrossValidation("RF")