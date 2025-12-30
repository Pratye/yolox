#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

"""
MPS-compatible DataPrefetcher.

For MPS devices, CUDA streams are not available, so we use a simpler
prefetching mechanism without streams.
"""

import torch


class DataPrefetcherMPS:
    """
    Simple data prefetcher for MPS devices (no CUDA streams).
    
    This is a simplified version that works on MPS by moving data
    to device synchronously.
    """
    
    def __init__(self, loader, device="mps"):
        self.loader = iter(loader)
        self.device = device
        self.preload()
    
    def preload(self):
        """Preload next batch."""
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
            # Move to device synchronously (MPS doesn't support async)
            self.next_input = self.next_input.to(self.device, non_blocking=False)
            self.next_target = self.next_target.to(self.device, non_blocking=False)
        except StopIteration:
            self.next_input = None
            self.next_target = None
    
    def next(self):
        """Get next batch."""
        input = self.next_input
        target = self.next_target
        self.preload()  # Preload next batch
        return input, target

