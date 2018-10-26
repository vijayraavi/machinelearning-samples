using BikeSharingDemand.Helpers;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Text;

namespace BikeSharingDemand
{
    public static class BikeSharingDataPreprocessor
    {
        public static IEstimator<ITransformer> DataPreprocessPipeline => _dataPreprocessPipeline;
        private static IEstimator<ITransformer> _dataPreprocessPipeline;

        private static string[] _featureColumns = new[] {
            "Season", "Year", "Month",
            "Hour", "Holiday", "Weekday",
            "Weather", "Temperature", "NormalizedTemperature",
            "Humidity", "Windspeed" };

        static BikeSharingDataPreprocessor()
        {
            //Configure data transformations in the Preprocess pipeline
            var mlContext = new MLContext();
            _dataPreprocessPipeline =
                // Copy the Count column to the Label column
                new CopyColumnsEstimator(mlContext, "Count", "Label")
                    // Concatenate all the numeric columns into a single features column
                    .Append(new ConcatEstimator(mlContext, "Features", _featureColumns));
        }
    }
}
