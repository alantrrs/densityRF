#include <Data.h>

using namespace MicrosoftResearch::Cambridge::Sherwood;

class HistogramAggregator{
    std::vector<float> weights_;
    std::vector<unsigned int> bins_;
    unsigned int sampleCount_;
  public:
    double Entropy() const;
    double weightedEntropy() const;
    HistogramAggregator();
    HistogramAggregator(int nClasses);
    HistogramAggregator(const std::vector<float>& weights);
    float GetProbability(int classIndex) const;
    int BinCount(){ return bins_.size();}
    unsigned int SampleCount() const { return sampleCount_; }
    int FindTallestBinIndex() const;
    // IStatisticsAggregator implementation
    void Clear();
    void Aggregate(const IDataPointCollection& data, unsigned int index);
    void Aggregate(const HistogramAggregator& aggregator);
    HistogramAggregator DeepClone() const;
};

