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

            // 1. Common data and data pre-processing
            var trainingData = BikeSharingDataReader.Read(TrainingDataLocation);
            var testData = BikeSharingDataReader.Read(TestDataLocation);
            var dataPreprocessingPipeline = BikeSharingDataReader.DataPreprocessingPipeline;
            var mlContext = new MLContext();


            var regressionLearner = new (string name, IEstimator<ITransformer> value)[]
            {
                ("FastTree", mlContext.Regression.Trainers.FastTree("Label", "Features")),
                ("SDCA", mlContext.Regression.Trainers.StochasticDualCoordinateAscent("Label", "Features")),
                ("Poisson", mlContext.Regression.Trainers.PoissonRegression("Label", "Features")),
                //Other possible learners that could be included
                //...FastForestRegressor...
                //...OnlineGradientDescentRegressor...
                //...FastTreeTweedieRegressor...
                //...GeneralizedAdditiveModelRegressor...
            };


            // Train, evaluate and test each of the regression trainers above
            foreach (var learner in regressionLearner)
            {
                var modelBuilder = new ModelBuilder(dataPreprocessingPipeline, learner.value);
                modelBuilder.Train(trainingData);
                modelBuilder.TestSinglePrediction();
                var metrics = modelBuilder.Evaluate(testData);
                modelBuilder.PrintRegressionMetrics($"{learner.name} regression model", metrics);
            }

            
            // 4. Visualize some predictions compared to observations from the test dataset

            //var fastTreeTester = new ModelTester<RegressionPredictionTransformer<FastTreeRegressionPredictor>>();
            //fastTreeTester.VisualizeSomePredictions("Fast Tree regression model", TestDataLocation, fastTreeModel, 10);

            //var sdcaTester = new ModelTester<RegressionPredictionTransformer<LinearRegressionPredictor>>();
            //sdcaTester.VisualizeSomePredictions("SDCA regression model", TestDataLocation, sdcaModel, 10);

            //var poissonTester = new ModelTester<RegressionPredictionTransformer<PoissonRegressionPredictor>>();
            //poissonTester.VisualizeSomePredictions("Poisson regression model", TestDataLocation, poissonModel, 10);

            //// 5. Just saving as .ZIP file the model based on Fast Tree which is the one with better accuracy and tests
            //modelBuilder.SaveModelAsFile(fastTreeModel, @".\FastTreeModel.zip");

            Console.ReadLine();
        }
    }
}
