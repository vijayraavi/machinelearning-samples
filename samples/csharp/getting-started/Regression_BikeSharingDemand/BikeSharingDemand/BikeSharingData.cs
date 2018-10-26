using Microsoft.ML.Runtime.Api;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BikeSharingDemand
{
    public class BikeSharingData
    {
        public class Prediction
        {
            [ColumnName("Score")]
            public float PredictedCount;
        }

        public class DemandSample
        {
            public float Season;
            public float Year;
            public float Month;
            public float Hour;
            public float Holiday;
            public float Weekday;
            public float WorkingDay;
            public float Weather;
            public float Temperature;
            public float NormalizedTemperature;
            public float Humidity;
            public float Windspeed;
            public float Count;   // This is the observed count, to be used a "label" to predict
        }

        public static List<DemandSample> ReadCsv(string dataLocation)
        {
            // Since bike demand data fits in memory, we can load it all in memory by
            // using ToList() at the end. This makes the processing more efficient.
            // For larger dataset, the data can be read as IEnumerable instead.
            return File.ReadLines(dataLocation)
                .Skip(1)
                .Select(x => x.Split(','))
                .Select(x => new DemandSample()
                {
                    Season = float.Parse(x[2]),
                    Year = float.Parse(x[3]),
                    Month = float.Parse(x[4]),
                    Hour = float.Parse(x[5]),
                    Holiday = float.Parse(x[6]),
                    Weekday = float.Parse(x[7]),
                    WorkingDay = float.Parse(x[8]),
                    Weather = float.Parse(x[9]),
                    Temperature = float.Parse(x[10]),
                    NormalizedTemperature = float.Parse(x[11]),
                    Humidity = float.Parse(x[12]),
                    Windspeed = float.Parse(x[13]),
                    Count = float.Parse(x[16])
                }).ToList();
        }
    }
}
