import DatasetPreparation
import MlpConstruction
import GBRconstruction


if __name__ == '__main__':
    # DatasetPreparation.kFoldSplit(10)
    # MlpConstruction.kFoldCrossValidation()
    # MlpConstruction.adaptiveBoostingMLP()
    # MlpConstruction.evaluateMLP()
    # GBRconstruction.gridSearchGradientRegressor()
    GBRconstruction.kFoldGradientBoostingRegressor()
