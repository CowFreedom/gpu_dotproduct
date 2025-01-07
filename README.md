# Dot Product

This is the official source code for [my blog post](https://cowfreedom.de/#dot_product/introduction/) on common pitfalls designing high performance mathematical routines on the GPU and CPU.
It consists of multiple implementations of calculating a dot product for very large vectors.

## Introduction

While graphics professing units have deservedly found astounding success in many general parallel programming applications in recent years, one must not become fixated by them and try to brute complicated
solutions for problems that are better served by other approaches.

In [the blog post accompanying this repository](https://cowfreedom.de/#dot_product/introduction/) it is shown how a discrete GPU approach loses out to a vectorized CPU version in an almost embarassingly parallel task. In fact, it is investigated that on many computers the GPU version has no hope of catching up to the CPU approach due bandwith limits of PCI-Express itself. Nevertheless, the code in this repostiory serves as an approachable example that the expected results depend on the underlying hardware and proper analysis of all components is required to implement the right approach.