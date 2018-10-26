using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using BikeSharingDemand.BikeSharingDemandData;
using BikeSharingDemand.Helpers;
using BikeSharingDemand.Model;
using Microsoft.ML;
using Microsoft.ML.Core.Data;

namespace BikeSharingDemand
{
    internal static class Program
    {
        private static string TrainingDataLocation = @"Data/hour_train.csv";
        private static string TestDataLocation = @"Data/hour_test.csv";

        static void Main(string[] args)
        {
            // 1. (CESARDL) Common data and data pre-processing
            var trainingDataView = BikeSharingDataLoader.GetDataView(TrainingDataLocation);
            var testDataView = BikeSharingDataLoader.GetDataView(TestDataLocation);
            var dataPreprocessPipeline = BikeSharingDataPreprocessor.DataPreprocessPipeline;

            // 1. (OLDER - DataLoad and Transformations are coupled) Common data and data pre-processing
            //var trainingDataView = BikeSharingDataReader.Read(TrainingDataLocation);
            //var testDataView = BikeSharingDataReader.Read(TestDataLocation);
            //var dataPreprocessPipeline = BikeSharingDataReader.DataPreprocessPipeline;

            //Peek data in training DataView after applying the PreprocessPipeline's transformations  
            ConsoleHelper.PeekDataViewInConsole(trainingDataView, dataPreprocessPipeline, 10);
            ConsoleHelper.PeekFeaturesColumnDataInConsole("Features", trainingDataView, dataPreprocessPipeline, 10);

            var mlContext = new MLContext();

            var regressionLearners = new (string name, IEstimator<ITransformer> value)[]
            {
                ("FastTree", mlContext.Regression.Trainers.FastTree("Label", "Features")),
                //("OnlineGradientDescent", mlContext.Regression.Trainers.OnlineGradientDescent("Label", "Features")),
                ("Poisson", mlContext.Regression.Trainers.PoissonRegression("Label", "Features")),
                ("SDCA", mlContext.Regression.Trainers.StochasticDualCoordinateAscent("Label", "Features"))
                //Other possible learners that could be included
                //...FastForestRegressor...
                //...FastTreeTweedieRegressor...
                //...GeneralizedAdditiveModelRegressor...
            };

            // Per each regression trainer, Train, Evaluate, Test and Save a different model
            foreach (var learner in regressionLearners)
            {
                // Train the model
                var modelBuilder = new ModelBuilder(dataPreprocessPipeline, learner.value);
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
