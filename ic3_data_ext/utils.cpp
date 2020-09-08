#ifndef UTILS_CPP
#define UTILS_CPP

template <typename T>
class MeanVarianceAccumulator{
    // adopted from:
    // http://www.nowozin.net/sebastian/blog/streaming-mean-and-variance-computation.html
    public:
        T sumw = 0.0;
        T wmean = 0.0;
        T t = 0.0;
        int n = 0;

        void add_element(const T& value, const T& weight){
            assert(weight >= 0.0);

            T q = value - this->wmean;
            T temp_sumw = this->sumw + weight;
            T r = q*weight / temp_sumw;

            this->wmean += r;
            this->t += q*r*this->sumw;
            this->sumw = temp_sumw;
            this->n += 1;
        }

        T count(){
            return this->n;
        }

        T mean(){
            return this->wmean;
        }

        T var(){
            return (this->t * this->n)/( this->sumw*( this->n - 1));
        }

        T std(){
            return sqrt(this->var());
        }
};

#endif