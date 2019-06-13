#ifndef ALGORITHMS_H
#define ALGORITHMS_H

namespace Math
{

constexpr float epsilon = 1e-6;

template <typename T>
bool fuzzyCompareEqual(const T left, const T right);

template <>
bool fuzzyCompareEqual<float>(const float, const float);

} // namespace Math

#endif // ALGORITHMS_H
