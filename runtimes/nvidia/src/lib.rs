//! Nvidia GPU 运行时。

#![cfg(detected_cuda)]

use graph_topo::GraphTopo;

mod driver;
mod graph;
