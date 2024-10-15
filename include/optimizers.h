/*
 *
 *  Created on: May 31, 2023
 *      Author: kardon
 */

#ifndef DECISION_ENGINE_OPTIMIZERS_H_
#define DECISION_ENGINE_OPTIMIZERS_H_

#include "matrix.h"
#include "utils.h"

//fnv1a hash fwd decl:
uint64_t fnv1a(const std::string &str);

namespace provallo {
	
	//Filters : mean, median, max, bloom
	namespace filters 
	{

		class base_filter
		{
		public:
			virtual ~base_filter() {}
			virtual void filter(matrix<double>& m) = 0;
		};
		class mean_filter : public base_filter
		{
			size_t _window;
		public:
			mean_filter(size_t window) : _window(window) {}
			void filter(matrix<double>& m) override
			{
				for (size_t i = 0; i < m.rows(); ++i)
				{
					for (size_t j = 0; j < m.cols(); ++j)
					{
						double sum = 0.0;
						for (size_t k = 0; k < _window; ++k)
						{
							sum += m(i, j + k);
						}
						m(i, j) = sum / _window;
					}
				}
			}
		};	
		
		class median_filter : public base_filter
		{
			size_t _window;	
		public:
			median_filter(size_t window) : _window(window) {}
			void filter(matrix<double>& m) override
			{
				for (size_t i = 0; i < m.rows(); ++i)
				{
					for (size_t j = 0; j < m.cols(); ++j)
					{
						std::vector<double> tmp;
						for (size_t k = 0; k < _window; ++k)
						{
							tmp.push_back(m(i, j + k));
						}
						std::sort(tmp.begin(), tmp.end());
						m(i, j) = tmp[tmp.size() / 2];
					}
				}
			}
		};	
		class max_filter : public base_filter
		{
			size_t _window;
		public:
			max_filter(size_t window) : _window(window) {}
			void filter(matrix<double>& m) override
			{
				for (size_t i = 0; i < m.rows(); ++i)
				{
					for (size_t j = 0; j < m.cols(); ++j)
					{
						double max = std::numeric_limits<double>::min();
						for (size_t k = 0; k < _window; ++k)
						{
							if (m(i, j + k) > max)
							{
								max = m(i, j + k);
							}
						}
						m(i, j) = max;
					}
				}
			}
		};	
		//bloom filter
		class bloom_filter : public base_filter
		{
			size_t _window; //window size
			std::function<uint64_t(const std::string&)> _hash;
			std::vector<bool> _bloom;
			
		public:
			bloom_filter(size_t window) : _window(window),_hash(fnv1a),_bloom(window) {}
			void filter (const std::vector<std::string> &v)
			{
				for (auto e : v)
				{
					uint64_t h = _hash(e);
					_bloom[h % _bloom.size()] = true;
				}
			}
			void filter(matrix<double>& m) override
			{
				for (size_t i = 0; i < m.rows(); ++i)
				{
					for (size_t j = 0; j < m.cols(); ++j)
					{
						double sum = 0.0;
						for (size_t k = 0; k < _window; ++k)
						{
							sum += m(i, j + k);
						}
						m(i, j) = sum / _window;
					}
				}
			}
		};

		 

		

	}//namespace filters
	//quasi newton optimization:
	namespace QN
	{
		// Quasi-Newton methods
		// BFGS
		// DFP
		// Broyden
		// SR1

		struct bfgs_hessian
		{
			template <typename Array, typename Matrix>
			static Matrix
			inverse(const Array &y, const Array &dir, const Matrix &ihessian)
			{
				typedef typename Array::value_type float_type;
				float_type ytx = dot(y, dir);
				Matrix auxmat = identity<float_type>(dir.size()) - (1 / ytx) * (y * transpose(dir));
				return transpose(auxmat) * (ihessian * auxmat) + (1 / ytx) * (dir * transpose(dir));
			}
		};
		// DFP
		struct dfp_hessian
		{
			template <typename Array, typename Matrix>
			static Matrix
			inverse(const Array &y, const Array &dir, const Matrix &ihessian)
			{
				typedef typename Array::value_type float_type;
				float_type ytx = dot(y, dir);
				float_type ythy = dot(y, ihessian * y);
				Matrix auxmat = y * (transpose(y) * transpose(ihessian));
				return ihessian + (1 / ytx) * (dir * transpose(dir)) - (1 / ythy) * (ihessian * auxmat);
			}
		};
		// Broyden
		struct broyden_hessian
		{
			template <typename Array, typename Matrix>
			static Matrix
			inverse(const Array &y, const Array &dir, const Matrix &ihessian)
			{
				typedef typename Array::value_type float_type;
				float_type xthy = dot(dir, ihessian * y);
				Matrix auxmat = transpose(dir) * ihessian;
				return ihessian + (1 / xthy) * (dir - ihessian * y) * auxmat;
			}
		};
		// Broyden family
		template <typename Discriminator>
		struct broyden_family_hessian : public Discriminator
		{
			template <typename Array, typename Matrix>
			static Matrix
			inverse(const Array &y, const Array &dir, const Matrix &ihessian)
			{
				return (1 - Discriminator::phi()) * bfgs_hessian::inverse(y, dir, ihessian) + Discriminator::phi() * dfp_hessian::inverse(y, dir, ihessian);
			}
		};
		// SR1
		
		struct sr1_hessian
		{
			template <typename Array, typename Matrix>
			static Matrix
			inverse(const Array &y, const Array &dir, const Matrix &ihessian)
			{
				Array auxvect = dir - ihessian * y;
				return ihessian + (1 / dot(auxvect, y)) * auxvect * transpose(auxvect);
			}
		};
		// SR1 family
		template <typename Discriminator>
		struct sr1_family_hessian : public Discriminator
		{
			template <typename Array, typename Matrix>
			static Matrix
			inverse(const Array &y, const Array &dir, const Matrix &ihessian)
			{
				return (1 - Discriminator::phi()) * bfgs_hessian::inverse(y, dir, ihessian) + Discriminator::phi() * sr1_hessian::inverse(y, dir, ihessian);
			}
		};

		// Define default policies for operators
		class default_operators
		{
		public:
			typedef wolfe_linear_search linear_search;
			typedef bfgs_hessian hessian_estimator;
		};
		// Class to define a use of the default policy values.
		// Avoids ambiguities if we derive from Default*Operators more than once
		class default_operator_args : virtual public default_operators
		{
		};

		// ----- Quasi-Newton operators

		// Create helper classes to set the operators
		template <typename Operator>
		class linear_search_is : virtual public default_operators
		{
		public:
			typedef Operator LinearSearch; // overriding typedef
		};

		template <typename Operator>
		class hessian_estimator_is : virtual public default_operators
		{
		public:
			typedef Operator HessianEstimator; // overriding typedef
		};

		template <typename Base, int num>
		class Discriminator : public Base
		{
		};

		template <typename LinearSearchSetter, typename HessianEstimatorSetter>
		class OperatorSelector : public Discriminator<LinearSearchSetter, 1>,
								 public Discriminator<HessianEstimatorSetter, 2>
		{
		};

		
		
		
		
		// Quasi-Newton algorithm. Line search and hessian estimation are policies of the algorithm
		template <typename Array, typename Function,
				  typename LinearSearchSetter = default_operator_args,
				  typename HessianEstimatorSetter = default_operator_args>
		class quasi_newton
		{
			// float type of this class
			typedef typename Array::value_type float_type;
			typedef typename Array::size_type size_type;

			// Helper template to refer to the various operators
			typedef OperatorSelector<LinearSearchSetter, HessianEstimatorSetter> Operators;

			// Functor object
			Function _function;
			// Gradient of the function
			Gradient<Function, Array> _gradient;
			// Best current point
			Array _best;
			// Current alpha
			float_type _alpha;
			// Current inverse hessian
			matrix<float_type> _ihessian;
			// Current search direction
			Array _dir;
			// Current gradient direction
			Array _current_gradient;

		public:
			// Constructor

			quasi_newton(const Function &function, const Array &best) : _function(function), _gradient(function, 1E-04), _best(best), _alpha(
																																		  1.0),
																		_ihessian(best.size(), best.size(), float_type())
			{
				// Initialize the inverse hessian
				for (size_type i = 0; i < _best.size(); ++i)
					_ihessian(i, i) = 1;
				// Initial search direction
				_current_gradient = _gradient(_best);
				_dir = (-1.0) * (_ihessian * _current_gradient);
			}
			// Accessors

			const Array &
			best() const
			{
				return _best;
			}

			const Array &
			gradient() const
			{
				return _current_gradient;
			}

			void
			step()
			{
				// Calculate fitness
				float_type fx = _function(_best);

				// Get alpha
				_alpha = Operators::linear_search::alpha(_alpha, _best, _dir,
														 _function, fx, _gradient,
														 _current_gradient);
				// Move point
				_best = _best + _alpha * _dir;

				// New gradient (save old one for further calculation)
				Array old_gradient(_current_gradient);
				_current_gradient = _gradient(_best);

				// Calculate gradient difference
				Array y = _current_gradient - old_gradient;

				// Update inverse hessian matrix
				_ihessian = Operators::hessian_estimator::inverse(y, _dir,
																  _ihessian);

				// Calculate new search direction
				_dir = (-1.0) * (_ihessian * _current_gradient);
			}
			//
			// Perform a number of steps
			void
			steps(size_type n)
			{
				for (size_type i = 0; i < n; ++i)
					step();
			}
			//
			// Perform a number of steps
			void
			steps(size_type n, std::ostream &out)
			{
				for (size_type i = 0; i < n; ++i)
				{
					step();
					out << _best << std::endl;
				}
			}
			~quasi_newton()
			{
			}
		};
	} // QN

	namespace NM
	{

		// Initialize array
		template <class T, std::size_t N, class Function>
		void
		initArray(fixed_array<T, N> *values, const Function &function)
		{
		}

		template <class T, class Function>
		void
		initArray(std::vector<T> *values, const Function &function)
		{
			values->resize(function.nvars());
		}

		struct default_initializer
		{
			template <class Function, class Array>
			static void
			initialize(const Function &function, Array *values,
					   std::random_device &random)
			{
				initArray(values, function);
				std::mt19937 gen(random);

				std::uniform_int_distribution<> uniform(0, RAND_MAX);

				for (size_t i = 0; i < values->size(); ++i)
					(*values)[i] = function.left(i) + (function.right(i) - function.left(i)) * uniform(gen);
			}
		};

		struct less_is_better
		{
			template <class Pair>
			inline bool
			operator()(const Pair &a, const Pair &b) const
			{
				return (a.first < b.first);
			}
		};

		struct high_is_better
		{
			template <class Pair>
			inline bool
			operator()(const Pair &a, const Pair &b) const
			{
				return (a.first > b.first);
			}
		};

		// Define default policies for population operators
		class default_operators
		{
		public:
			typedef default_initializer Initializer;
			typedef less_is_better Comparator;
		};

		// Class to define a use of the default policy values.
		// Avoids ambiguities if we derive from Default*Operators more than once
		class default_operator_args : virtual public default_operators
		{
		};

		// ----- Nelder-Mead operators

		// Create helper classes to set the operators
		template <typename Operator>
		class initializer_is : virtual public default_operators
		{
		public:
			typedef Operator Initializer; // overriding typedef
		};

		template <typename Operator>
		class comparator_is : virtual public default_operators
		{
		public:
			typedef Operator Comparator; // overriding typedef
		};

		// ----- Operator selectors

		// This class allows having the same base class more than once
		template <typename Base, int num>
		class Discriminator : public Base
		{
		};

		template <typename InitializerSetter, typename ComparatorSetter>
		class OperatorSelector : public Discriminator<InitializerSetter, 1>,
								 public Discriminator<ComparatorSetter, 2>
		{
		};


		
		template <typename Array, typename Function,
				  typename InitializerSetter = default_operator_args,
				  typename ComparatorSetter = default_operator_args>
		class NelderMead
		{
			typedef typename Array::value_type float_type;

			// Helper template to refer to the various operators
			typedef OperatorSelector<InitializerSetter, ComparatorSetter> Operators;

			// Objective function
			Function _function;
			// Random number generator
			std::random_device _random;
			// Current points on the simplex
			std::vector<Array> _points;
			// Temporary buffer for sorted points
			typedef std::pair<float_type, Array *> PointType;
			std::vector<PointType> _sort_points;

			// Best current point
			Array _best;

			// Current centroid
			Array _centroid;

			// Reflection parameter
			float_type _alpha;
			// Expansion parameter
			float_type _beta;
			// Contraction parameter
			float_type _gamma;
			// Shrink parameter
			float_type _delta;

			// Method to initialize class after points initialization
			void
			init()
			{
				// Initialize sort points array
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
				for (size_t i = 0; i < _points.size(); ++i)
					_sort_points[i] = std::make_pair(_function(_points[i]),
													 &_points[i]);

				// Sort values
				std::sort(_sort_points.begin(), _sort_points.end(),
						  typename Operators::Comparator());

				// Compute centroid
				_centroid = *_sort_points[0].second;
				for (size_t i = 1; i < (_sort_points.size() - 1); ++i)
					_centroid = _centroid + *_sort_points[i].second;

				_centroid = (1.0 / (float_type)(_sort_points.size() - 1)) * _centroid;

				// Select best point
				_best = *_sort_points[0].second;
			}

		public:
			NelderMead(const Function &function, const std::random_device &random) : _function(function), _random(random), _points(
																															   _function.nvars() + 1),
																					 _sort_points(_function.nvars() + 1), _alpha(
																															  1.0),
																					 _beta(1.0 + 2.0 / _function.nvars()), _gamma(
																															   0.75 - 1.0 / (2.0 * _function.nvars())),
																					 _delta(
																						 1.0 - (1.0 / _function.nvars()))
			{

				// Initialize variables
				for (size_t i = 0; i < _points.size(); ++i)
					Operators::Initializer::initialize(_function, &_points[i],
													   _random);

				init();
			}

			NelderMead(const Function &function, const Array &point,
					   float_type radius, const std::random_device &rand) : _function(function), _random(rand), _points(_function.nvars() + 1), _sort_points(_function.nvars() + 1), _alpha(1.0), _beta(1.0 + 2.0 / _function.nvars()), _gamma(0.75 - 1.0 / (2.0 * _function.nvars())), _delta(1.0 - (1.0 / _function.nvars()))
			{
				// Random spherical point generator
				SphericalPoint<Array> sph(function.nvars(), radius);

				// Initialize variables
				_points[0] = point;
				for (size_t i = 1; i < _points.size(); ++i)
					_points[i] = point + sph(_random);

				init();
			}

			// Performs a Nelder-Mead step
			void
			step()
			{
				// Size (number of points)
				size_t num = _sort_points.size();

				// Important point
				float_type f1 = _sort_points[0].first;		  // Best
				float_type fn = _sort_points[num - 2].first;  // Second worst
				float_type fn1 = _sort_points[num - 1].first; // Worst

				// Compute reflection point
				Array xr = _centroid + _alpha * (_centroid - *_sort_points[num - 1].second);
				float_type fr = _function(xr);

				// Reference to worst pair
				PointType &worst = _sort_points[num - 1];

				if (f1 <= fr && fr < fn)
				{
					// Replace worst point
					*worst.second = xr;
					worst.first = fr;
				}
				else if (fr < f1)
				{
					// Compute expansion point
					Array xe = _centroid + _beta * (xr - _centroid);
					float_type fe = _function(xe);

					// Replace worst point
					if (fe < fr)
					{
						*worst.second = xe;
						worst.first = fe;
					}
					else
					{
						*worst.second = xr;
						worst.first = fr;
					}
				}
				else if (fn <= fr && fr < fn1)
				{
					// Compute outside contraction
					Array xoc = _centroid + _gamma * (xr - _centroid);
					float_type foc = _function(xoc);

					// Replace worst point
					if (foc < fr)
					{
						*worst.second = xoc;
						worst.first = foc;
					}
					else
					{
						// Shrink
						const Array &x1 = *_sort_points[0].second;
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
						for (size_t i = 1; i < _sort_points.size(); ++i)
						{
							*_sort_points[i].second = x1 + _delta * (*_sort_points[i].second - x1);
							_sort_points[i].first = _function(
								*_sort_points[i].second);
						}
					}
				}
				else
				{ // (fr >= fn1)
					// Compute inside contraction
					Array xic = _centroid - _gamma * (xr - _centroid);
					float_type fic = _function(xic);

					// Replace worst point
					if (fic < fn1)
					{
						*worst.second = xic;
						worst.first = fic;
					}
					else
					{
						// Shrink
						const Array &x1 = *_sort_points[0].second;
#ifdef HAVE_OPENMP
#pragma omp parallel for
#endif
						for (size_t i = 1; i < _sort_points.size(); ++i)
						{
							*_sort_points[i].second = x1 + _delta * (*_sort_points[i].second - x1);
							_sort_points[i].first = _function(
								*_sort_points[i].second);
						}
					}
				}

				// Sort values
				std::sort(_sort_points.begin(), _sort_points.end(),
						  typename Operators::Comparator());

				// Compute centroid
				_centroid = *_sort_points[0].second;
				for (size_t i = 1; i < (_sort_points.size() - 1); ++i)
					_centroid = _centroid + *_sort_points[i].second;

				_centroid = (1.0 / (float_type)(_sort_points.size() - 1)) * _centroid;

				// Select best point
				_best = *_sort_points[0].second;
			}

			// Get current centroid
			Array
			centroid() const
			{
				return _centroid;
			}

			// Get best individual on the current population
			Array
			best() const
			{
				return _best;
			}

			// Get reference fitness functor (from here the state of the functor can be safely changed)
			Function &
			getFitness()
			{
				return _function;
			}

			// Get constant reference to the fitness functor
			const Function &
			getFitness() const
			{
				return _function;
			}

			// Iterators to chromosomes on the population
			typedef typename std::vector<Array>::iterator iterator;
			typedef typename std::vector<Array>::const_iterator const_iterator;

			iterator
			begin()
			{
				return _points.begin();
			}

			const_iterator
			begin() const
			{
				return _points.begin();
			}

			iterator
			end()
			{
				return _points.end();
			}

			const_iterator
			end() const
			{
				return _points.end();
			}

			const float_type &
			getReflection() const
			{
				return _alpha;
			}

			void
			setReflection(const float_type &alpha)
			{
				_alpha = alpha;
			}

			const float_type &
			getExpansion() const
			{
				return _beta;
			}

			void
			setExpansion(const float_type &beta)
			{
				_beta = beta;
			}

			const float_type &
			getContraction() const
			{
				return _delta;
			}

			void
			setContraction(const float_type &delta)
			{
				_delta = delta;
			}

			const float_type &
			getShrink() const
			{
				return _gamma;
			}

			void
			setShrink(const float_type &gamma)
			{
				_gamma = gamma;
			}

			~NelderMead()
			{
			}
		};

	} // NM
	namespace GENETIC
	{

		struct less_is_better
		{
			template <class Chromosome>
			inline bool
			operator()(Chromosome a, Chromosome b) const
			{
				return (a.fitness < b.fitness);
			}
		};

		// Simple class to compare two chromosomes ("high is better" means that the sort algorithm will put the best
		// chromosome with the bigger fitness on the first element of the sorted container)
		struct high_is_better
		{
			template <class Chromosome>
			inline bool
			operator()(Chromosome a, Chromosome b) const
			{
				return (a.fitness > b.fitness);
			}
		};
		template <typename chromo_type>
		struct chromosome
		{
			chromo_type *chromosome;
			double fitness;
			chromosome(chromo_type *achromosome = 0, double afitness = 0.0) : chromosome(achromosome), fitness(afitness)
			{
			}
		};

		struct ga_warning
		{
			ga_warning(const std::string &message)
			{
				std::cerr << "[%]GENETIC Warning : " << message << std::endl;
			}
		};

		struct default_initializer
		{
			// Default initializer (don't do anything)
			template <class Fitness, class T>
			static void
			initialize(const Fitness &fitness, T *value,
					   std::random_device &random)
			{
				static ga_warning warning(
					"Using default initializer for chromosome of type " + std::string(typeid(T).name()));
			}
		};

		struct default_scaling
		{
			// Default scaling (don't do anything)
			template <class OutputIterator>
			static void
			scale(OutputIterator begin, OutputIterator end)
			{
				static ga_warning warning("Using default scaling");
			}
		};

		struct default_mutator
		{
			// Default mutator (don't do anything)
			template <class Fitness, class T, class GaData>
			static void
			mutate(const Fitness &fitness, T *value, std::random_device &random,
				   const GaData &ga_data)
			{
				static ga_warning warning(
					"Using default mutator for chromosome of type " + std::string(typeid(T).name()));
			}
		};

		struct default_sexcrossover
		{
			// Default crossover (like father, like son ; like mother, like daughter)
			template <class Fitness, class T, class GaData>
			static void
			crossover(const Fitness &fitness, const T &mom, const T &dad, T *son,
					  T *daughter, std::random_device &random,
					  const GaData &ga_data)
			{
				static ga_warning warning(
					"Using default sexual crossover for chromosome of type " + std::string(typeid(T).name()));
				*son = dad;
				*daughter = mom;
			}
		};
		struct elitist_selector
		{
			// Elitism rate
			static real_t _elitism_rate;

			// Default selector (roulette wheel selector)
			// Population : population class
			// OutputIterator : Iterator to an array of chromosomes
			template <class InputIterator, class OutputIterator, class Comparator>
			static void
			select(std::random_device &random, InputIterator ibegin,
				   InputIterator iend, OutputIterator begin, Comparator comp)
			{
				// Size of the population
				size_t npop(iend - ibegin);
				// Sort fitness
				std::sort(ibegin, iend, comp);
				// Number of best parents
				size_t nelite(_elitism_rate * npop);
				// Sanity check
				if ((nelite^(~nelite)) == 0)
					nelite = 1;
				// Put the best parents into the buffer
				for (size_t i = 0; i < npop; ++i)
				{
					(*begin++) = *((*(ibegin + i % nelite)).chromosome);
				}
			}
		};

		// Roulette wheel selector using the rank of the individual
		struct rank_selector
		{

			// Initialize sampler container
			static inline std::vector<float>
			initSampler(size_t npop)
			{
				std::vector<float> values(npop);
				float total_sum(0.0);
				for (size_t i = 0; i < npop; ++i)
				{
					float value = (npop - i) * (npop - i) * (npop - i) * (npop - i) * (npop - i);
					total_sum += value;
					if (i > 0)
						values[i] = value + values[i - 1];
					else
						values[i] = value;
				}
				for (size_t i = 0; i < npop; ++i)
					values[i] /= total_sum;

				return values;
			}

			template <class InputIterator, class OutputIterator, class Comparator>
			static void
			select(std::random_device &random, InputIterator ibegin,
				   InputIterator iend, OutputIterator begin, Comparator comp)
			{
				// Size of the population
				static std::vector<float> fitness(initSampler(iend - ibegin));
				std::mt19937 gen(random);
				std::uniform_real_distribution<float> uniform(0.0, 1.0);
				// std::uniform_int_distribution<> uniform(0,RAND_MAX);

				size_t npop(iend - ibegin);
				// Sort fitness
				std::sort(ibegin, iend, comp);
				// Sample each individual
				for (size_t i = 0; i < npop; ++i)
				{
					float rho = uniform(gen);
					size_t idx = std::lower_bound(fitness.begin(), fitness.end(),
												  rho) -
								 fitness.begin();
					(*begin++) = *((*(ibegin + idx)).chromosome);
				}
			}
		};
		// Define default policies for population operators
		class default_operators
		{
		public:
			typedef default_initializer Initializer;
			typedef elitist_selector Selector;
			typedef default_scaling Scaling;
			typedef default_sexcrossover Crossover;
			typedef default_mutator Mutator;
			typedef less_is_better Comparator;
		};

		class default_operator_args : virtual public default_operators
		{
		};

		// Create helper classes to set the operators
		template <typename Operator>
		class initializer_is : virtual public default_operators
		{
		public:
			typedef Operator Initializer; // overriding typedef
		};

		template <typename Operator>
		class selector_is : virtual public default_operators
		{
		public:
			typedef Operator Selector; // overriding typedef
		};

		template <typename Operator>
		class scaling_is : virtual public default_operators
		{
		public:
			typedef Operator Scaling; // overriding typedef
		};

		template <typename Operator>
		class comparator_is : virtual public default_operators
		{
		public:
			typedef Operator Comparator; // overriding typedef
		};

		template <typename Operator>
		class pop_operators : virtual public default_operators
		{
		public:
			typedef Operator Initializer; // overriding typedef
			typedef Operator Fitness;	  // overriding typedef
		};

		// ----- Genetic operators

		// Create helper classes to set the operators
		template <typename Operator>
		class crossover_is : virtual public default_operators
		{
		public:
			typedef Operator Crossover; // overriding typedef
		};

		template <typename Operator>
		class mutator_is : virtual public default_operators
		{
		public:
			typedef Operator Mutator; // overriding typedef
		};

		template <typename Operator>
		class gen_operators : virtual public default_operators
		{
		public:
			typedef Operator Crossover; // overriding typedef
			typedef Operator Mutator;	// overriding typedef
		};

		template <typename Base, int num>
		class Discriminator : public Base
		{
		};

		template <typename InitializerSetter, typename SelectorSetter,
				  typename ScalingSetter, typename ComparatorSetter>
		class PopOperatorSelector : public Discriminator<InitializerSetter, 1>,
									public Discriminator<SelectorSetter, 2>,
									public Discriminator<ScalingSetter, 4>,
									public Discriminator<ComparatorSetter, 5>
		{
		};

		template <typename InitializerSetter, typename SelectorSetter,
				  typename ScalingSetter, typename ComparatorSetter,
				  typename CrossoverSetter, typename MutatorSetter>
		class GenOperatorSelector : public Discriminator<InitializerSetter, 1>,
									public Discriminator<SelectorSetter, 2>,
									public Discriminator<ScalingSetter, 4>,
									public Discriminator<ComparatorSetter, 5>,
									public Discriminator<CrossoverSetter, 6>,
									public Discriminator<MutatorSetter, 7>
		{
		};

	}//namespace GENETIC

	namespace global
	{

		struct global_params
		{
			// Maximum number of steps on individual's life when is adapting to the environment
			static size_t max_life_steps;
			// Mutation rate of individuals
			static double mutation_rate;
			// Crossover rate of the population
			static double crossover_rate;
			// Max tolerance on gradient (an individual will stop growing if the gradient is lower than this value)
			static double tolerance;
			// Maximum number of iterations before convergence
			static size_t max_iterations;
			// Maximum number of GA best individual repetitions
			static size_t max_repetitions;
		};

		// GA data that could be used inside operators
		class GaData
		{
			// Time (i.e. iteration number)
			size_t _time;

			public:
			GaData() : _time(0)
			{
			}

			// Get time
			size_t
			time() const
			{
				return _time;
			}

			// Add delta to time value
			void
			addTime(size_t delta)
			{
				_time += delta;
			}

			// Set time value
			void
			setTime(size_t value)
			{
				_time = value;
			}

			~GaData()
			{
			}
		};

		// Check values
		template <class Function, class Array>
		static void
		checkValues(const Function &function, Array *values)
		{
			// Check interval limits
			for (size_t i = 0; i < values->size(); ++i)
			{
				if ((*values)[i] < function.left(i))
					(*values)[i] = function.left(i);
				else if ((*values)[i] > function.right(i))
					(*values)[i] = function.right(i);
			}
		}

		// Initialize array
		template <class T, std::size_t N, class Function>
		void
		initArray(fixed_array<T, N> *values, const Function &function)
		{
		}

		template <class T, class Function>
		void
		initArray(std::vector<T> *values, const Function &function)
		{
			values->resize(function.nvars());
		}

		// ------ Genetic Algorithm operators for the real value function
		struct Initializer
		{
			template <class Function, class Array>
			static void
			initialize(const Function &function, Array *values,
					   std::random_device &random)
			{
				initArray(values, function);
				std::mt19937 gen(random);
				std::uniform_int_distribution<> uniform(0, RAND_MAX);

				for (size_t i = 0; i < values->size(); ++i)
					(*values)[i] = function.left(i) + (function.right(i) - function.left(i)) * uniform(gen);
			}
		};

		struct NelderMeadMutator
		{

			// Calculate centroid (including worst point)
			template <class Array, class Function>
			static Array
			calculateCentroid(const NM::NelderMead<Array, Function> &nm)
			{
				using namespace NM;
				typedef typename NelderMead<Array, Function>::const_iterator ConstNmIt;
				Array centroid = *nm.begin();
				size_t num(nm.end() - nm.begin());
				for (ConstNmIt it = nm.begin() + 1; it != nm.end(); ++it)
					centroid = centroid + (*it);
				return (1.0 / num) * centroid;
			}
		};
			struct LifeMutator
			{
				template <class Function, class Array>
				static void
				mutate(const Function &function, Array *values,
					   std::random_device &random, const GaData &ga_data)
				{
					Array distances(*values);
					for (size_t i = 0; i < distances.size(); ++i)
						distances[i] = std::min(
							fabs((*values)[i] - function.left(i)),
							fabs((*values)[i] - function.right(i)));
					typename Array::value_type radius = *std::min_element(
						distances.begin(), distances.end());

					// After mutation, make the individual to grow up and be mature enough
					using namespace NM;
					NelderMead<Array, Function> nm(function, *values, radius);

					Array old_best(nm.best());
					for (size_t i = 0; i < global_params::max_life_steps; ++i)
					{
						nm.step();
						Array centroid = NelderMeadMutator::calculateCentroid<Array, Function>(nm);
						Array diff = nm.best() - centroid;
						if (sqrt(dot(diff, diff)) < global_params::tolerance)
							break;
					}

					//avoid nan	
					if (not isNan(nm.best()))
						*values = nm.centroid();

					checkValues<Function>(function, values);
				}
			};

			struct FinalMutator
			{
				template <class Function, class Array>
				static void
				mutate(const Function &function, Array *values,
					   std::random_device &random, const GaData &ga_data)
				{
					Array distances(*values);
					for (size_t i = 0; i < distances.size(); ++i)
						distances[i] = std::min(
							fabs((*values)[i] - function.left(i)),
							fabs((*values)[i] - function.right(i)));
					typename Array::value_type radius = *std::min_element(
						distances.begin(), distances.end());

					// After mutation, make the individual to grow up and be mature enough
					NM::NelderMead<Array, Function> nm(function, *values, radius);

					Array old_best(nm.best());
					for (size_t i = 0; i < global_params::max_iterations; ++i)
					{
						nm.step();
						Array centroid = NelderMeadMutator::calculateCentroid<Array, Function>(nm);
						Array diff = nm.best() - centroid;
						if (sqrt(dot(diff, diff)) < global_params::tolerance)
							break;
					}

					// Update value
					*values = nm.best();

					checkValues<Function>(function, values);
				}
			};//struct FinalMutator
	
	 
		}//namespace global
			
	
	}//namespace provallo
	 
 
#endif /* DECISION_ENGINE_OPTIMIZERS_H_ */
