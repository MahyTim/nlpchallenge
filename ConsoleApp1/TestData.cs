using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BinaryClassification_SentimentAnalysis
{
    public static class TestData
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);

        public static string PrepareTestDataAndReturnPath(params int[] additionalDataSets)
        {
            var orginalFilePath = Path.Combine(AppPath, "datasets", "data.csv");
            var targetFilePath = Path.Combine(AppPath, "datasets", "preprocessed.csv");
            if (File.Exists(targetFilePath))
            {
                File.Delete(targetFilePath);
            }
            using (var targetFile = File.Open(targetFilePath, FileMode.CreateNew))
            using (var targetFileWriter = new StreamWriter(targetFile))
            {
                var lines = File.ReadLines(orginalFilePath);
                foreach (var line in lines)
                {
                    var parts = line.Split(',');
                    if (parts.Length == 1)
                    {
                        Console.WriteLine("ERROR in file: " + parts[0]);
                    }
                    else
                    {
                        var label = parts[0];
                        if (false == int.TryParse(label, out int x))
                        {
                            Console.WriteLine("ERROR in file : " + label);
                        }
                        var text = parts[1];
                        var preprocessedLines = PreprocessAndReturnLineOrNull(label, text);
                        foreach (var preprocessedLine in preprocessedLines)
                        {
                            if (false == string.IsNullOrWhiteSpace(preprocessedLine))
                            {
                                targetFileWriter.WriteLine(preprocessedLine);
                            }
                        }
                    }
                }
            }

            var paths = new List<string>();
            paths.Add(targetFilePath);
            var additionalPaths = PreprocessAdditionalDataSetsAndReturnPaths(additionalDataSets).ToArray();
            paths.AddRange(additionalPaths);
            paths.Add(Path.Combine(AppPath, "datasets", "classifications.csv"));
            return string.Join('+', paths);
        }

        private static IEnumerable<string> PreprocessAdditionalDataSetsAndReturnPaths(int[] additionalDataSets)
        {
            if (additionalDataSets != null)
            {
                foreach (var additionalDataSet in additionalDataSets)
                {
                    var orginalFilePath = Path.Combine(AppPath, "datasets", $"{additionalDataSet}.csv");
                    if (false == File.Exists(orginalFilePath))
                    {
                        Console.WriteLine("Path does not exist: " + orginalFilePath);
                    }
                    var targetFilePath = Path.Combine(AppPath, "datasets", $"{additionalDataSet}_preprocessed.csv");
                    if (File.Exists(targetFilePath))
                    {
                        File.Delete(targetFilePath);
                    }
                    using (var targetFile = File.Open(targetFilePath, FileMode.CreateNew))
                    using (var targetFileWriter = new StreamWriter(targetFile))
                    {
                        var lines = File.ReadLines(orginalFilePath);
                        foreach (var line in lines)
                        {
                            targetFileWriter.WriteLine($"{additionalDataSet},{line}".ToLowerInvariant());
                        }
                    }
                    yield return targetFilePath;
                }
            }
        }

        private static string[] BlockWords = new[] { "basiscursus", "postgraduaat", "-daagse", "graduaat", "opleiding", "inleiding" };

        private static IEnumerable<string> PreprocessAndReturnLineOrNull(string label, string text)
        {
            // 1. alles naar lowercase
            text = text.ToLowerInvariant();
            // 2. bepaalde woorden die geen context bieden verwijderen
            //foreach (var blockWord in BlockWords)
            //{
            //    text = text.Replace(blockWord, string.Empty);
            //}
            //// Domein specifieke
            // Alles netjes trimmen nog
            text = text.Trim();
            yield return string.Join(',', label, text);
        }
    }
}