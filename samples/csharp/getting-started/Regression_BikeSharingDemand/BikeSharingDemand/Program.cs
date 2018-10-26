using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using BikeSharingDemand.Helpers;
using BikeSharingDemand.BikeSharingModel;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;

namespace BikeSharingDemand
{
    internal static class Program
    {
        private static string TrainingDataLocation = @"Data/hour_train.csv";
        private static string TestDataLocation = @"Data/hour_test.csv";

        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // 1. Common data and data pre-processing
            var trainingDataView = mlContext.CreateDataView(BikeSharingData.ReadCsv(TrainingDataLocation));
            var testDataView = mlContext.CreateDataView(BikeSharingData.ReadCsv(TestDataLocation));
            var dataPreprocessor = new BikeSharingDataPreprocessor(mlContext);
            var dataPreprocessPipeline = dataPreprocessor.DataPreprocessPipeline;

            //Peek data in training DataView after applying the PreprocessPipeline's transformations  
            ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataPreprocessPipeline, 10);
            ConsoleHelper.PeekFeaturesColumnDataInConsole(mlContext, "Features", trainingDataView, dataPreprocessPipeline, 10);

            var regressionLearners = new (string name, IEstimator<ITransformer> value)[]
            {
                ("FastTree", mlContext.Regression.Trainers.FastTree()),
                ("OnlineGradientDescent", mlContext.Regression.Trainers.OnlineGradientDescent()),
                ("Poisson", mlContext.Regression.Trainers.PoissonRegression()),
                ("SDCA", mlContext.Regression.Trainers.StochasticDualCoordinateAscent())
                //Other possible learners that could be included
                //...FastForestRegressor...
                //...FastTreeTweedieRegressor...
                //...GeneralizedAdditiveModelRegressor...
            };

            // Per each regression trainer, Train, Evaluate, Test and Save a different model
            foreach (var learner in regressionLearners)
            {
                // Train the model
                var modelBuilder = new BikeSharingModel(dataPreprocessPipeline, learner.value);
                var trainedModel = modelBuilder.Train(trainingDataView);

                //Test single prediction
                modelBuilder.TestSinglePrediction();

                //Evaluate model's accuracy
                var metrics = modelBuilder.Evaluate(testDataView);
                modelBuilder.PrintRegressionMetrics($"{learner.name} regression model", metrics);

                //Visualize 10 tests comparing prediction with actual/observed values from the test dataset
                ModelTester.VisualizeSomePredictions(learner.name, TestDataLocation, trainedModel, 10);

                //Save the model file that can be used by any application
                modelBuilder.SaveModelAsFile($"./{learner.name}Model.zip");
            }

            Console.ReadLine();
        }
    }
}
