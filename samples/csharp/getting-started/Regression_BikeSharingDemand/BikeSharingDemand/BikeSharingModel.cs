using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace BikeSharingDemand
{
    public class BikeSharingModel
    {
        private MLContext _mlcontext;
        private IEstimator<ITransformer> _trainingPipeline;

        public ITransformer TrainedModel { get; private set; }
        public PredictionFunction<BikeSharingData.DemandSample, BikeSharingData.Prediction> PredictionFunction { get; private set;}

        public BikeSharingModel(
            IEstimator<ITransformer> dataPreprocessPipeline,
            IEstimator<ITransformer> regressionLearner)
        {
            _mlcontext = new MLContext();
            _trainingPipeline = dataPreprocessPipeline.Append(regressionLearner);
        }
        
        public ITransformer Train(IDataView trainingData)
        {
            Console.WriteLine("=============== Training model ===============");
            TrainedModel = _trainingPipeline.Fit(trainingData);
            PredictionFunction = TrainedModel.MakePredictionFunction<BikeSharingData.DemandSample, BikeSharingData.Prediction>(_mlcontext);
            return TrainedModel;
        }

        public void TestSinglePrediction()      
        {
            CheckTrained();

            //Sample: 
            // instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
            // 13950,2012-08-09,3,1,8,10,0,4,1,1,0.8,0.7576,0.55,0.2239,72,133,205
            var demandSample = new BikeSharingData.DemandSample()
            {
                Season = 3,
                Year = 1,
                Month = 8,
                Hour = 10,
                Holiday = 0,
                Weekday = 4,
                WorkingDay = 1,
                Weather = 1,
                Temperature = (float)0.8,
                NormalizedTemperature = (float)0.7576,
                Humidity = (float)0.55,
                Windspeed = (float)0.2239
            };

            var prediction = PredictionFunction.Predict(demandSample);
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"Predicted : {prediction.PredictedCount}");
            Console.WriteLine($"*************************************************");
        }

        public RegressionEvaluator.Result Evaluate(IDataView testData)
        {
            CheckTrained();
            Console.WriteLine("=============== Evaluating Model's accuracy with Test data===============");
            var predictions = TrainedModel.Transform(testData);
            var metrics = _mlcontext.Regression.Evaluate(predictions, "Count", "Score");
            return metrics;
        }

        public void PrintRegressionMetrics(string name, RegressionEvaluator.Result metrics)
        {
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Metrics for {name}          ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       LossFn: {metrics.LossFn:0.##}");
            Console.WriteLine($"*       R2 Score: {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Absolute loss: {metrics.L1:#.##}");
            Console.WriteLine($"*       Squared loss: {metrics.L2:#.##}");
            Console.WriteLine($"*       RMS loss: {metrics.Rms:#.##}");
            Console.WriteLine($"*************************************************");
        }

        public void SaveModelAsFile(string persistedModelPath)
        {
            CheckTrained();
            using (var fs = new FileStream(persistedModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                _mlcontext.Model.Save(TrainedModel, fs);
            Console.WriteLine("The model is saved to {0}", persistedModelPath);
        }

        private void CheckTrained()
        {
            if (TrainedModel == null || PredictionFunction == null)
                throw new InvalidOperationException("Cannot test before training. Call Train() first.");
        }

    }
}
