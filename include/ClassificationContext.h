#include <StatisticalAggregators.h>

using namespace MicrosoftResearch::Cambridge::Sherwood;

template<class F>
class ClassificationContext : public ITrainingContext<F,HistogramAggregator> {
  private:
    int maxOffset_;
    std::vector<float> weights_;
  public:
    ClassificationContext(const ImageCollection& trainingData) {
      maxOffset_ = trainingData.getMargin();
      weights_ = std::vector<float>(trainingData.CountClasses(),0.0);
      for (int i=0;i<trainingData.Count();i++){
        weights_[trainingData.getLabel(i)]+=1;
      }
      for (int c=0;c<weights_.size();c++){
        weights_[c]=(float)trainingData.Count()/weights_[c];
      }
    }
    // Implementation of ITrainingContext
    F GetRandomFeature(Random& random){
      cv::Point p1(random.Next(-maxOffset_,maxOffset_),random.Next(-maxOffset_,maxOffset_));
      cv::Point p2(random.Next(-maxOffset_,maxOffset_),random.Next(-maxOffset_,maxOffset_));
      int channel = random.Next(0,3);
      return PixelCompFeature(p1,p2,channel);
    }

    HistogramAggregator GetStatisticsAggregator(){
      return HistogramAggregator(weights_);
    }
    double ComputeInformationGain(const HistogramAggregator& allStatistics, const HistogramAggregator& leftStatistics, const HistogramAggregator& rightStatistics){
      double entropyBefore = allStatistics.weightedEntropy();
      unsigned int nTotalSamples = leftStatistics.SampleCount() + rightStatistics.SampleCount();
      if (nTotalSamples <= 1)
        return 0.0;
      double le = leftStatistics.weightedEntropy();
      double re = rightStatistics.weightedEntropy();
      double entropyAfter = (leftStatistics.SampleCount() * le + rightStatistics.SampleCount() * re) / nTotalSamples;
     return entropyBefore - entropyAfter;
    }
    bool ShouldTerminate(const HistogramAggregator& parent, const HistogramAggregator& leftChild, const HistogramAggregator& rightChild, double gain){
      return gain < 0.01;
    }
};
