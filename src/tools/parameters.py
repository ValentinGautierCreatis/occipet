#!/usr/bin/env python3

class Parameters:
    def __init__(self, data: dict) -> None:
        self.data = data

    def __getitem__(self, key):
        return self.data.get(key)
