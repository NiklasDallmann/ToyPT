#ifndef ALGORITHMS_H
#define ALGORITHMS_H

namespace Math
{

constexpr double epsilon = 1e-6;

template <typename T>
bool fuzzyCompareEqual(const T left, const T right);

template <>
bool fuzzyCompareEqual<double>(const double, const double);

} // namespace Math

#endif // ALGORITHMS_H
