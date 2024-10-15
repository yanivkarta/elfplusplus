#ifndef __UTILS_H__
#define __UTILS_H__

/*
 * utils.h
 *
 *  Created on: Mar 22, 2023
 *      Author: kardon
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <random>
#include <functional>
#include <vector>
#include <string>
#include <bits/unique_ptr.h>
#include <cmath>

#include <cstddef>

// use std::pow instead of pow to avoid ambiguity

#ifndef real_t
#define real_t double /* supported: float, double */
#endif
#ifndef sparse_ix
#define sparse_ix int64_t /* supported: int, int64_t, size_t */
#endif

// #define square(x) ((x)*(x))
#define likely(x) __builtin_expect((bool)(x), true)
#define unlikely(x) __builtin_expect((bool)(x), false)
#define THRESHOLD_EXACT_S 87670 /* difference is <5e-4 */
#define pow2(n) (((size_t)1) << (n))
#define div2(n) ((n) >> 1)
#define mult2(n) ((n) << 1)
#define ix_parent(ix) (div2((ix) - (size_t)1)) /* integer division takes care of deciding left-right */
#define ix_child(ix) (mult2(ix) + (size_t)1)
#define SD_MIN 1e-10
#define ix_comb_(i, j, n, ncomb) (((ncomb) + ((j) - (i))) - (size_t)1 - div2(((n) - (i)) * ((n) - (i) - (size_t)1)))
#define ix_comb(i, j, n, ncomb) (((i) < (j)) ? ix_comb_(i, j, n, ncomb) : ix_comb_(j, i, n, ncomb))
#define calc_ncomb(n) (((n) % 2) == 0) ? (div2(n) * ((n) - (size_t)1)) : ((n) * div2((n) - (size_t)1))
#define THRESHOLD_LONG_DOUBLE (size_t)1e6
#define RNG_engine std::mt19937_64
#define UniformUnitInterval std::uniform_real_distribution<double>
#define hashed_set std::unordered_set
#define hashed_map std::unordered_map
#define is_na_or_inf(x) (std::isnan(x) || std::isinf(x))
// pendantic mode
#define UNDEF_REFERENCE(x)                         \
    ptrdiff_t var_unreference = (ptrdiff_t)(&(x)); \
    var_unreference++;
#define UNDEF_REFERENCE2(x)              \
    var_unreference = (ptrdiff_t)(&(x)); \
    var_unreference++;
#define set_return_position(x) NULL
#define return_to_position(x, y)
namespace provallo
{
#if SIZE_MAX == UINT32_MAX /* 32-bit systems */

    constexpr static const uint32_t MultiplyDeBruijnBitPosition[32] =
        {0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28,
         15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31};

    size_t
    log2ceil(size_t v)
    {
        v--;
        v |= v >> 1; // first round down to one less than a power of 2
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;

        return MultiplyDeBruijnBitPosition[(uint32_t)(v * 0x07C4ACDDU) >> 27] + 1;
    }
#elif SIZE_MAX == UINT64_MAX /* 64-bit systems */
    constexpr static const uint64_t tab64[64] =
        {63, 0, 58, 1, 59, 47, 53, 2, 60, 39, 48, 27, 54, 33, 42, 3, 61, 51, 37,
         40, 49, 18, 28, 20, 55, 30, 34, 11, 43, 14, 22, 4, 62, 57, 46, 52, 38,
         26, 32, 41, 50, 36, 17, 19, 29, 10, 13, 21, 56, 45, 25, 31, 35, 16, 9,
         12, 44, 24, 15, 8, 23, 7, 6, 5};

    inline size_t
    log2ceil(size_t value)
    {
        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        value |= value >> 32;
        return tab64[((uint64_t)((value - (value >> 1)) * 0x07EDD5E59A4E28C2)) >> 58] + 1;
    }
#endif

#define EULERS_GAMMA 0.577215664901532860606512

    template <typename T>
    struct loss
    {
        typedef T value_type;
        std::function<value_type(value_type, value_type)> loss_func;
        std::function<value_type(value_type, value_type)> loss_grad;
        std::function<value_type(value_type, value_type)> loss_hess;
        loss(std::function<T(T, T)> loss_func_, std::function<T(T, T)> loss_grad_, std::function<T(T, T)> loss_hess_) : loss_func(loss_func_), loss_grad(loss_grad_), loss_hess(loss_hess_)
        {
        }

        T operator()(T x, T y) const
        {
            return loss_func(x, y);
        }

        T grad(T x, T y) const
        {
            return loss_grad(x, y);
        }
        T hess(T x, T y) const
        {
            return loss_hess(x, y);
        }
        // apply loss function on container
        template <typename T1>
        T1 apply(T1 x, T1 y) const
        {
            T1 res(x);
            for (size_t i = 0; i < x.size1(); i++)
                for (size_t j = 0; j < x.size2(); j++)
                    res(i, j) = loss_func(x(i, j), y(i, j));
            return res;
        }
    };
    // kullback-leibler divergence loss
    template <class T>
    struct kl_loss : public loss<T>
    {
        kl_loss() : loss<T>(
                        kl_loss<T>::loss_func,
                        kl_loss<T>::grad,
                        kl_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y));
        }
        static T grad(T x, T y)
        {
            return (log((1 - x) / (1 - y)) - log(x / y));
        }
        static T hess(T x, T y)
        {
            x = x;
            return (1 / (y * (1 - y)) + 1 / ((1 - y) * y));
        }
    };
    // cross-entropy loss
    template <class T>
    struct ce_loss : public loss<T>
    {
        ce_loss() : loss<T>(ce_loss<T>::loss_func, ce_loss<T>::grad, ce_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return -x * log(y) - (1 - x) * log(1 - y);
        }
        static T grad(T x, T y)
        {
            return (y - x) / (y * (1 - y));
        }
        static T hess(T x, T y)
        {
            return (y - x) / ((y * y) * (1 - y));
        }
    };
    // hinge loss
    template <class T>
    struct hinge_loss : public loss<T>
    {
        hinge_loss() : loss<T>(hinge_loss<T>::loss_func, hinge_loss<T>::grad, hinge_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return std::max(0, 1 - x * y);
        }
        static T grad(T x, T y)
        {
            return (x * y < 1) ? -y : 0;
        }
        static T hess(T x, T y)
        {
            return 0 * ((x + y) / (y + x));
        }
    };
    // huber loss
    template <class T>
    struct huber_loss : public loss<T>
    {
        huber_loss() : loss<T>(huber_loss<T>::loss_func, huber_loss<T>::grad, huber_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return (x - y) * (x - y) / 2;
        }
        static T grad(T x, T y)
        {
            return x - y;
        }
        static T hess(T x, T y)
        {
            return 1 * ((x + y) / (y + x));
        }
    };
    // logistic loss
    template <class T>
    struct logistic_loss : public loss<T>
    {
        logistic_loss() : loss<T>(logistic_loss<T>::loss_func, logistic_loss<T>::grad, logistic_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return log(1 + exp(-x * y));
        }
        static T grad(T x, T y)
        {
            return -y / (1 + exp(x * y));
        }
        static T hess(T x, T y)
        {
            return y * y * exp(x * y) / ((1 + exp(x * y)) * (1 + exp(x * y)));
        }
    };
    // modified huber loss
    template <class T>
    struct modified_huber_loss : public loss<T>
    {
        modified_huber_loss() : loss<T>(modified_huber_loss<T>::loss_func, modified_huber_loss<T>::grad, modified_huber_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return (x * y < 1) ? (1 - x * y) * (1 - x * y) : 0;
        }
        static T grad(T x, T y)
        {
            return (x * y < 1) ? -2 * y * (1 - x * y) : 0;
        }
        static T hess(T x, T y)
        {
            return (x * y < 1) ? 2 * y * y : 0;
        }
    };
    // quantile loss
    template <class T>
    struct quantile_loss : public loss<T>
    {
        quantile_loss() : loss<T>(quantile_loss<T>::loss_func, quantile_loss<T>::grad, quantile_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return (x - y) * (x - y) / 2;
        }
        static T grad(T x, T y)
        {
            return (x - y);
        }
        static T hess(T x, T y)
        {
            return 1 * ((x + y) / (y + x));
        }
    };
    // squared loss
    template <class T>
    struct squared_loss : public loss<T>
    {
        squared_loss() : loss<T>(squared_loss<T>::loss_func, squared_loss<T>::grad, squared_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return (x - y) * (x - y) / 2;
        }
        static T grad(T x, T y)
        {
            return (x - y);
        }
        static T hess(T x, T y)
        {
            // avoid warning
            return 1 * ((x + y) / (y + x));
        }
    };
    // smoothed hinge loss
    template <class T>
    struct smoothed_hinge_loss : public loss<T>
    {
        smoothed_hinge_loss() : loss<T>(smoothed_hinge_loss<T>::loss_func, smoothed_hinge_loss<T>::grad, smoothed_hinge_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return std::max(0, 1 - x * y);
        }
        static T grad(T x, T y)
        {
            return (x * y < 1) ? -y : 0;
        }
        static T hess(T x, T y)
        {
            return 0.0 * x * y;
        }
    };
    // squared hinge loss
    template <class T>
    struct squared_hinge_loss : public loss<T>
    {
        squared_hinge_loss() : loss<T>(squared_hinge_loss<T>::loss_func, squared_hinge_loss<T>::grad, squared_hinge_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return std::max(0, 1 - x * y);
        }
        static T grad(T x, T y)
        {
            return (x * y < 1) ? -y : 0;
        }
        static T hess(T x, T y)
        {
            // avoid warning
            return 0.0 * x * y;
        }
    };
    // welsch loss
    template <class T>
    struct welsch_loss : public loss<T>
    {
        welsch_loss() : loss<T>(welsch_loss<T>::loss_func, welsch_loss<T>::grad, welsch_loss<T>::hess)
        {
        }
        static T loss_func(T x, T y)
        {
            return 1 - exp(-(x - y) * (x - y) / 2);
        }
        static T grad(T x, T y)
        {
            return (x - y) * exp(-(x - y) * (x - y) / 2);
        }
        static T hess(T x, T y)
        {
            return (1 - (x - y) * (x - y)) * exp(-(x - y) * (x - y) / 2);
        }
    };

    // loss functions for regression

    // splitting criterion

    extern "C"
    {
        typedef enum NewCategAction
        {
            Weighted = 0,
            Smallest = 11,
            Random = 12
        } NewCategAction; /* Weighted means Impute in the extended model */
        typedef enum MissingAction
        {
            Divide = 21,
            Impute = 22,
            Fail = 0
        } MissingAction; /* Divide is only for non-extended model */
        typedef enum ColType
        {
            Numeric = 31,
            Categorical = 32,
            NotUsed = 0
        } ColType;
        typedef enum CategSplit
        {
            SubSet = 0,
            SingleCateg = 41
        } CategSplit;
        typedef enum CoefType
        {
            Uniform = 61,
            Normal = 0
        } CoefType; /* For extended model */
        typedef enum UseDepthImp
        {
            Lower = 71,
            Higher = 0,
            Same = 72
        } UseDepthImp; /* For NA imputation */
        typedef enum WeighImpRows
        {
            Inverse = 0,
            Prop = 81,
            Flat = 82
        } WeighImpRows; /* For NA imputation */
        typedef enum ScoringMetric
        {
            Depth = 0,
            Density = 92,
            BoxedDensity = 94,
            BoxedDensity2 = 96,
            BoxedRatio = 95,
            AdjDepth = 91,
            AdjDensity = 93
        } ScoringMetric;

        typedef enum ColCriterion
        {
            Uniformly = 0,
            ByRange = 1,
            ByVar = 2,
            ByKurt = 3
        } ColCriterion; /* For proportional choices */
        typedef enum GainCriterion
        {
            NoCrit = 0,
            Averaged = 1,
            Pooled = 2,
            FullGain = 3,
            DensityCrit = 4
        } Criterion; /* For guided splits */

    }; //  compatible with isolation model
    // for imputation 
    

    inline void
    unexpected_error()
    {
        std::cerr << "error" << errno << std::endl;
    }
   
    template <typename T, size_t N>
    T distance_squared(const std::array<T, N> &point_a,
                       const std::array<T, N> &point_b)
    {
        T d_squared = T();
        for (typename std::array<T, N>::size_type i = 0; i < N; ++i)
        {
            auto delta = point_a[i] - point_b[i];
            d_squared += delta * delta;
        }
        return d_squared;
    }

    template <typename T, size_t N>
    T distance(const std::array<T, N> &point_a, const std::array<T, N> &point_b)
    {
        return std::sqrt(distance_squared(point_a, point_b));
    }

    template <typename T, size_t N>
    std::vector<T>
    closest_distance(const std::vector<std::array<T, N>> &means,
                     const std::vector<std::array<T, N>> &data)
    {
        std::vector<T> distances;
        distances.reserve(data.size());
        for (auto &d : data)
        {
            T closest = distance_squared(d, means[0]);
            for (auto &m : means)
            {
                T distance = distance_squared(d, m);
                if (distance < closest)
                    closest = distance;
            }
            distances.push_back(closest);
        }
        return distances;
    }
    // Random Plus Plus for k-means
    template <typename T, size_t N>
    std::vector<std::array<T, N>>
    random_plusplus(const std::vector<std::array<T, N>> &data, uint32_t k,
                    uint64_t seed)
    {
        using input_size_t = typename std::array<T, N>::size_type;
        std::vector<std::array<T, N>> means;
        // Using a very simple PRBS generator, parameters selected according to
        // https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
        std::linear_congruential_engine<uint64_t, 6364136223846793005,
                                        1442695040888963407, UINT64_MAX>
            rand_engine(seed);

        // Select first mean at random from the set
        {
            std::uniform_int_distribution<input_size_t> uniform_generator(
                0, data.size() - 1);
            means.push_back(data[uniform_generator(rand_engine)]);
        }

        for (uint32_t count = 1; count < k; ++count)
        {
            // Calculate the distance to the closest mean for each data point
            auto distances = closest_distance(means, data);
            // Pick a random point weighted by the distance from existing means
            // TODO: This might convert floating point weights to ints, distorting the distribution for small weights
#if !defined(_MSC_VER) || _MSC_VER >= 1900
            std::discrete_distribution<input_size_t> generator(
                distances.begin(), distances.end());
#else // MSVC++ older than 14.0
            input_size_t i = 0;
            std::discrete_distribution<input_size_t> generator(distances.size(), 0.0, 0.0, [&distances, &i](double)
                                                               { return distances[i++]; });
#endif
            means.push_back(data[generator(rand_engine)]);
        }
        return means;
    }
    // k-means closest mean
    template <typename T, size_t N>
    uint32_t
    closest_mean(const std::array<T, N> &point,
                 const std::vector<std::array<T, N>> &means)
    {

        T smallest_distance = distance_squared(point, means[0]);
        typename std::array<T, N>::size_type index = 0;
        T distance;
        for (size_t i = 1; i < means.size(); ++i)
        {
            distance = distance_squared(point, means[i]);
            if (distance < smallest_distance)
            {
                smallest_distance = distance;
                index = i;
            }
        }
        return index;
    }
    // k-means calculate clusters
    template <typename T, size_t N>
    std::vector<uint32_t>
    calculate_clusters(const std::vector<std::array<T, N>> &data,
                       const std::vector<std::array<T, N>> &means)
    {
        std::vector<uint32_t> clusters;
        for (auto &point : data)
        {
            clusters.push_back(closest_mean(point, means));
        }
        return clusters;
    }

    // k-means calculate means
    template <typename T, size_t N>
    std::vector<std::array<T, N>>
    calculate_means(const std::vector<std::array<T, N>> &data,
                    const std::vector<uint32_t> &clusters,
                    const std::vector<std::array<T, N>> &old_means, uint32_t k)
    {
        std::vector<std::array<T, N>> means(k);
        std::vector<T> count(k, T());
        for (size_t i = 0; i < std::min(clusters.size(), data.size()); ++i)
        {
            auto &mean = means[clusters[i]];
            count[clusters[i]] += 1;
            for (size_t j = 0; j < std::min(data[i].size(), mean.size()); ++j)
            {
                mean[j] += data[i][j];
            }
        }
        for (size_t i = 0; i < k; ++i)
        {
            if (count[i] == 0)
            {
                means[i] = old_means[i];
            }
            else
            {
                for (size_t j = 0; j < means[i].size(); ++j)
                {
                    means[i][j] /= count[i];
                }
            }
        }
        return means;
    }

    // k-means deltas
    template <typename T, size_t N>
    std::vector<T>
    deltas(const std::vector<std::array<T, N>> &old_means,
           const std::vector<std::array<T, N>> &means)
    {
        std::vector<T> distances;
        distances.reserve(means.size());
        assert(old_means.size() == means.size());
        for (size_t i = 0; i < means.size(); ++i)
        {
            distances.push_back(distance(means[i], old_means[i]));
        }
        return distances;
    }
    // k-means deltas below limit
    template <typename T>
    bool
    deltas_below_limit(const std::vector<T> &deltas, T min_delta)
    {
        for (T d : deltas)
        {
            if (d > min_delta)
            {
                return false;
            }
        }
        return true;
    }

    // solve quadratic equation
    template <typename real_value>
    int
    solveQuadratic(real_value a, real_value b, real_value c, real_value *x0,
                   real_value *x1)
    {
        real_value disc = b * b - 4 * a * c;

        if (a == 0)
        {
            if (b == 0)
            {
                return 0;
            }
            else
            {
                *x0 = -c / b;
                return 1;
            };
        }

        if (disc > 0)
        {
            if (b == 0)
            {
                real_value r = fabs(0.5 * sqrt(disc) / a);
                *x0 = -r;
                *x1 = r;
            }
            else
            {
                real_value sgnb = (b > 0 ? 1 : -1);
                real_value temp = -0.5 * (b + sgnb * sqrt(disc));
                real_value r1 = temp / a;
                real_value r2 = c / temp;

                if (r1 < r2)
                {
                    *x0 = r1;
                    *x1 = r2;
                }
                else
                {
                    *x0 = r2;
                    *x1 = r1;
                }
            }
            return 2;
        }
        else if (disc == 0)
        {
            *x0 = -0.5 * b / a;
            *x1 = -0.5 * b / a;
            return 2;
        }
        else
        {
            return 0;
        }
    }

    // cubic interpolation
    template <typename real_value>
    real_value
    interpQuad(real_value f0, real_value fp0, real_value f1, real_value zl,
               real_value zh)
    {
        real_value fl = f0 + zl * (fp0 + zl * (f1 - f0 - fp0));
        real_value fh = f0 + zh * (fp0 + zh * (f1 - f0 - fp0));
        real_value c = 2 * (f1 - f0 - fp0);

        real_value zmin = zl, fmin = fl;

        if (fh < fmin)
        {
            zmin = zh;
            fmin = fh;
        }

        if (c > 0)
        {
            real_value z = -fp0 / c;
            if (z > zl && z < zh)
            {
                real_value f = f0 + z * (fp0 + z * (f1 - f0 - fp0));
                if (f < fmin)
                {
                    zmin = z;
                    fmin = f;
                };
            }
        }

        return zmin;
    }
    // cubic
    template <typename real_value>
    real_value
    cubic(real_value c0, real_value c1, real_value c2, real_value c3,
          real_value z)
    {
        return c0 + z * (c1 + z * (c2 + z * c3));
    }
    // check extremum
    template <typename real_value>
    void
    checkExtremum(real_value c0, real_value c1, real_value c2, real_value c3,
                  real_value z, real_value *zmin, real_value *fmin)
    {
        real_value y = cubic(c0, c1, c2, c3, z);
        if (y < *fmin)
        {
            *zmin = z;
            *fmin = y;
        }
    }
    // cubic interpolation
    template <typename real_value>
    real_value
    interpCubic(real_value f0, real_value fp0, real_value f1, real_value fp1,
                real_value zl, real_value zh)
    {
        real_value eta = 3 * (f1 - f0) - 2 * fp0 - fp1;
        real_value xi = fp0 + fp1 - 2 * (f1 - f0);
        real_value c0 = f0, c1 = fp0, c2 = eta, c3 = xi;
        real_value zmin, fmin;
        real_value z0, z1;

        zmin = zl;
        fmin = cubic(c0, c1, c2, c3, zl);
        checkExtremum(c0, c1, c2, c3, zh, &zmin, &fmin);

        int n = solveQuadratic(3 * c3, 2 * c2, c1, &z0, &z1);

        if (n == 2)
        {
            if (z0 > zl && z0 < zh)
                checkExtremum(c0, c1, c2, c3, z0, &zmin, &fmin);
            if (z1 > zl && z1 < zh)
                checkExtremum(c0, c1, c2, c3, z1, &zmin, &fmin);
        }
        else if (n == 1)
        {
            if (z0 > zl && z0 < zh)
                checkExtremum(c0, c1, c2, c3, z0, &zmin, &fmin);
        }

        return zmin;
    }

    // interpolation
    template <typename real_value>
    real_value
    interpolate(real_value a, real_value fa, real_value fpa, real_value b,
                real_value fb, real_value fpb, real_value xmin,
                real_value xmax, int order)
    {
        real_value z, alpha, zmin, zmax;

        zmin = (xmin - a) / (b - a);
        zmax = (xmax - a) / (b - a);

        if (zmin > zmax)
        {
            real_value tmp = zmin;
            zmin = zmax;
            zmax = tmp;
        };

        if (order > 2 && not isnan(fpb))
            z = interpCubic(fa, fpa * (b - a), fb, fpb * (b - a), zmin, zmax);
        else
            z = interpQuad(fa, fpa * (b - a), fb, zmin, zmax);

        alpha = a + z * (b - a);

        return alpha;
    }

    // line search
    struct wolfe_linear_search
    {
        template <typename Function, typename Gradient, typename Array>
        static typename Array::value_type
        alpha(typename Array::value_type alpha1, const Array &best,
              const Array &dir, const Function &function,
              typename Array::value_type fx, const Gradient &gradient,
              const Array &current_gradient)
        {
            typedef typename Array::value_type real_value;

            // Max number of iterations
            static const size_t bracket_iters = 100, section_iters = 100;

            // Recommended values from Fletcher are :
            static const real_value rho = 0.01;
            static const real_value sigma = 0.1;
            static const real_value tau1 = 9;
            static const real_value tau2 = 0.05;
            static const real_value tau3 = 0.5;

            real_value falpha, fpalpha, delta, alpha_next;
            real_value alpha = alpha1, alpha_prev = 0.0;

            // Initialize function values
            real_value f0 = fx;
            real_value fp0 = dot(current_gradient, dir);

            real_value a(0.0), b(alpha), fa(f0), fb(0.0), fpa(fp0), fpb(0.0);

            // Initialize previous values
            real_value falpha_prev = f0;
            real_value fpalpha_prev = fp0;

            // Temporary value
            Array temp;

            // Begin bracketing
            size_t i = 0;
            while (i++ < bracket_iters)
            {
                // Calculate function in alpha
                temp = best + alpha * dir;
                falpha = function(temp);

                // Fletcher's rho test
                if (falpha > f0 + alpha * rho * fp0 || falpha >= falpha_prev)
                {
                    a = alpha_prev;
                    fa = falpha_prev;
                    fpa = fpalpha_prev;
                    b = alpha;
                    fb = falpha;
                    fpb = NAN;
                    break;
                }

                fpalpha = dot(gradient(temp), dir);

                // Fletcher's sigma test
                if (fabs(fpalpha) <= -sigma * fp0)
                    return alpha;

                if (fpalpha >= 0)
                {
                    a = alpha;
                    fa = falpha;
                    fpa = fpalpha;
                    b = alpha_prev;
                    fb = falpha_prev;
                    fpb = fpalpha_prev;
                    break; // goto sectioning
                }

                delta = alpha - alpha_prev;

                real_value lower = alpha + delta;
                real_value upper = alpha + tau1 * delta;

                alpha_next = interpolate(alpha_prev, falpha_prev, fpalpha_prev,
                                         alpha, falpha, fpalpha, lower, upper, 3);

                alpha_prev = alpha;
                falpha_prev = falpha;
                fpalpha_prev = fpalpha;
                alpha = alpha_next;
            }

            while (i++ < section_iters)
            {
                delta = b - a;

                real_value lower = a + tau2 * delta;
                real_value upper = b - tau3 * delta;

                alpha = interpolate(a, fa, fpa, b, fb, fpb, lower, upper, 3);
                temp = best + alpha * dir;
                falpha = function(temp);

                if ((a - alpha) * fpa <= std::numeric_limits<real_value>::epsilon())
                {
                    // Roundoff prevents progress
                    return alpha;
                };

                if (falpha > f0 + rho * alpha * fp0 || falpha >= fa)
                {
                    //  a_next = a;
                    b = alpha;
                    fb = falpha;
                    fpb = NAN;
                }
                else
                {
                    fpalpha = dot(gradient(temp), dir);

                    if (fabs(fpalpha) <= -sigma * fp0)
                        return alpha; // terminate

                    if (((b - a) >= 0 && fpalpha >= 0) || ((b - a) <= 0 && fpalpha <= 0))
                    {
                        b = a;
                        fb = fa;
                        fpb = fpa;
                        a = alpha;
                        fa = falpha;
                        fpa = fpalpha;
                    }
                    else
                    {
                        a = alpha;
                        fa = falpha;
                        fpa = fpalpha;
                    }
                }
            }
            return alpha;
        }
    };

    // log(x) function
    template <int Base>
    inline double
    log(double x)
    {
        return ::log(x) / ::log(Base);
    }

    // x log (x) function
    template <int Base>
    inline double
    xlog(double x)
    {
        if (x == 0.0)
            return 0.0;
        return x * log<Base>(x);
    }

    // fnv1a hash function
    inline uint64_t
    fnv1a(const std::string &text)
    {
        constexpr const real_t fnv_prime = 16777619;
        constexpr const real_t fnv_offset_basis = 2166136261;

        uint64_t hash = fnv_offset_basis;
        for (size_t i = 0; i < text.size(); i++)
        {
            hash ^= text[i];
            hash *= fnv_prime;
        }
        return hash;
    }

    // wang hash function
    inline std::uint64_t wang64(const uint64_t &x)
    {
        std::uint64_t y(x);
        y = (~y) + (y << 21); // y = (y << 21) - y - 1;
        y = y ^ (y >> 24);
        y = (y + (y << 3)) + (y << 8); // y * 265
        y = y ^ (y >> 14);
        y = (y + (y << 2)) + (y << 4); // y * 21
        y = y ^ (y >> 28);
        y = y + (y << 31);
        return y;
    }

    // wang hash function
    inline uint32_t wang32(const uint32_t &x)
    {
        uint32_t y(x);
        y = (~y) + (y << 15); // y = (y << 15) - y - 1;
        y = y ^ (y >> 12);
        y = (y + (y << 2)) + (y << 4); // y * 21
        y = y ^ (y >> 9);
        y = (y + (y << 3)) + (y << 4); // y * 28
        y = y ^ (y >> 23);
        y = y + (y << 1) + (y << 4); // y * 21 * 5
        return y;
    }

    // wang hash function
    inline uint16_t wang16(const uint16_t &x)
    {
        uint16_t y(x);
        y = (~y) + (y << 7); // y = (y << 7) - y - 1;
        y = y ^ (y >> 4);
        y = (y + (y << 3)) + (y << 4); // y * 21
        y = y ^ (y >> 10);
        y = y + (y << 1) + (y << 4); // y * 21 * 5
        return y;
    }

    // check if number is even
    template <typename UIntType>
    constexpr bool is_even(const UIntType val)
    {
        return val & 1;
    }
    // check if number is pow2
    template <typename UIntType>
    inline bool is_pow2(const UIntType val)
    {
        if (val < 0)
        { // safety against signed ints.

            throw std::invalid_argument(std::string("error: is_pow2 argument ") + std::to_string(val) + " should be nonnegative.");
        }
        return !(val == 0) && !(val & (val - 1));
    }
    template <typename UIntType>
    inline UIntType next_pow2(const UIntType val)
    {
        if (val < 0)
        { // safety against signed ints.

            throw std::invalid_argument(std::string("error: next_pow2 argument ") + std::to_string(val) + " should be nonnegative.");
        }
        if (is_pow2(val))
        {
            return val;
        }
        UIntType next_pow2 = 1;
        while (next_pow2 < val)
        {
            next_pow2 <<= 1;
        }
        return next_pow2;
    }
    template <typename UIntType>
    inline UIntType prev_pow2(const UIntType val)
    {
        if (val < 0)
        { // safety against signed ints.

            throw std::invalid_argument(std::string("error: prev_pow2 argument ") + std::to_string(val) + " should be nonnegative.");
        }
        if (is_pow2(val))
        {
            return val;
        }
        UIntType prev_pow2 = 1;
        while (prev_pow2 < val)
        {
            prev_pow2 <<= 1;
        }
        return prev_pow2 >> 1;
    }
    template <typename UIntType>
    inline UIntType log2(const UIntType val)
    {
        if (val < 0)
        { // safety against signed ints.

            throw std::invalid_argument(std::string("error: log2 argument ") + std::to_string(val) + " should be nonnegative.");
        }
        if (val == 0)
        {
            return 0;
        }
        UIntType log2 = 0;
        UIntType value = val;
        while (value >>= 1)
        {
            ++log2;
        }
        return log2;
    }
    template <typename UIntType>
    inline UIntType log2_ceil(const UIntType val)
    {
        if (val < 0)
        { // safety against signed ints.

            throw std::invalid_argument(std::string("error: log2_ceil argument ") + std::to_string(val) + " should be nonnegative.");
        }
        if (val == 0)
        {
            return 0;
        }
        UIntType log2 = 0;
        UIntType value = val - 1;
        while (value >>= 1)
        {
            ++log2;
        }
        return log2 + 1;
    } // log2_ceil

    std::string
    trim(const std::string &pString, const std::string &pWhitespace);
    std::string
    reduce(const std::string &pString, const std::string &pFill,
           const std::string &pWhitespace = " ");
} // namespace
#endif