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
    # GBRconstruction.gridSearchGradientRegressor()
    # GBRconstruction.kFoldGradientBoostingRegressor()
    # MlpConstruction.testKModels()
    # BaseModels.kFoldCrossValidation("KNN")
    # BaseModels.kFoldCrossValidation("LR")
    # BaseModels.kFoldCrossValidation("SVM")
    BaseModels.kFoldCrossValidation("RF")