# procedural-framework

> Skeletal Singleton Tree (SST) system for procedural geometry generation

## Overview

The SST is a functional L-system architecture where:
- **The specification IS the execution**
- **The tree IS the program**
- **The algebra IS the geometry**

## Core Concepts

- **State/Mutation Separation** - Matrix stack (state) is orthogonal to geometry operations (mutation)
- **Compositional Algebra** - Operations compose functionally with deterministic output
- **Platform Agnostic** - Transpiles to Maya, Blender, UE5

## Documentation

- [Skeletal Singleton Tree](./docs/skeletal-singleton-tree.md) - Architecture blueprint
- [Node Algebra](./docs/node-algebra.md) - Functional alphabet and composition rules
- [Extended State Algebra](./docs/extended-state-algebra.md) - Phase 6: Spreads, Frames, Fields
