#!/usr/bin/env python3
import json

class Parameters:
    def __init__(self, data: dict) -> None:
        self.data = data


    def __getitem__(self, key):
        return self.data.get(key)


    def to_json(self, path : str) -> None:
        with open(path, 'w') as json_file:
            json.dump(self.data, json_file)
