using BikeSharingDemand.BikeSharingDemandData;
using BikeSharingDemand.Helpers;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using System;
using System.Linq;

namespace BikeSharingDemand.Model
{
    public static class ModelTester
    {
        public static void VisualizeSomePredictions(string modelName, string testDataLocation, ITransformer model, int numberOfPredictions)
        {
            //Prediction test
            var mlcontext = new MLContext();

            // Create prediction engine 
            var engine = model.MakePredictionFunction<BikeSharingDemandSample, BikeSharingDemandPrediction>(mlcontext);

            //Make the provided number of predictions and compare with observed data from the test dataset
            var testData = new BikeSharingDemandsCsvReader().GetDataFromCsv(testDataLocation).ToList();

            Console.WriteLine($"=======================================================");
            Console.WriteLine($"=======       Tests with {modelName}       ========");

            for (int i = 0; i < numberOfPredictions; i++)
            {
                var prediction = engine.Predict(testData[i]);
       
                Console.WriteLine($"-------------------------------------------------");
                Console.WriteLine($"Predicted : {prediction.PredictedCount}");
                Console.WriteLine($"Actual:     {testData[i].Count}");
                Console.WriteLine($"-------------------------------------------------");
            }

            Console.WriteLine($"=======================================================");
            Console.WriteLine();
        }
    }
}
