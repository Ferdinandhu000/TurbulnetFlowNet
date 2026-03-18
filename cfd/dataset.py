import os
import sys
from typing import List, Tuple, Dict, Any, Literal
import shutil
from tqdm import tqdm
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from cfd.sensors import LHS, AroundCylinder
from cfd.embedding import Voronoi, Mask, Vector


class DatasetMixin:

    def load2tensor(self, case_dir: str) -> torch.Tensor:
        # Check for single .npy file in the case directory
        npy_files = [f for f in os.listdir(case_dir) if f.endswith('.npy')]
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {case_dir}")
        
        data_path = os.path.join(case_dir, npy_files[0])
        data = torch.from_numpy(np.load(data_path)).cuda().float()
        
        # (B, H, W, C) -> (B, C, H, W)
        if data.ndim == 4:
            data = data.permute(0, 3, 1, 2)
        elif data.ndim == 3: # (H, W, C) -> (1, C, H, W)
            data = data.permute(2, 0, 1).unsqueeze(0)
            
        return data

    def prepare_sensor_timeframes(self, n_chunks: int) -> torch.IntTensor:
        # prepare sensor timeframes (fixed)
        sensor_timeframes: torch.Tensor = (
            torch.tensor(self.init_sensor_timeframes, device='cuda') + torch.arange(n_chunks, device='cuda').unsqueeze(1)
        )
        assert sensor_timeframes.shape == (n_chunks, len(self.init_sensor_timeframes))
        return sensor_timeframes.int()

    def prepare_fullstate_timeframes(
        self,
        n_chunks: int,
        seed: int | None = None,
        init_fullstate_timeframes: List[int] | None = None,
    ) -> torch.IntTensor:
        assert seed is not None or init_fullstate_timeframes is not None, 'must be either deterministic or random'
        if seed is None and init_fullstate_timeframes is not None:    # deterministic
            fullstate_timeframes: torch.Tensor = (
                torch.arange(n_chunks, device='cuda').unsqueeze(1) 
                + torch.tensor(init_fullstate_timeframes, device='cuda').unsqueeze(0)
            )
            assert fullstate_timeframes.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk)
            return fullstate_timeframes
        
        else:
            assert seed is not None, 'seed must be specified when target frames are generated randomly'
            fullstate_timeframes: torch.Tensor = torch.empty((n_chunks, self.n_fullstate_timeframes_per_chunk), dtype=torch.int, device='cuda')
            for chunk_idx in range(n_chunks):
                torch.random.manual_seed(seed + chunk_idx)

                if self.future_prediction_range is not None:
                    # New logic: sample from a future range
                    range_start = self.future_prediction_range[0]
                    range_end = self.future_prediction_range[1]
                    offset = max(self.init_sensor_timeframes)
                    range_size = range_end - range_start + 1
                    
                    random_init_timeframes = offset + range_start + torch.randperm(
                        n=range_size, device='cuda'
                    )[:self.n_fullstate_timeframes_per_chunk].sort()[0]

                else:
                    # Original logic: sample within the sensor range
                    random_init_timeframes: torch.Tensor = torch.randperm(
                        n=max(self.init_sensor_timeframes), device='cuda'
                    )[:self.n_fullstate_timeframes_per_chunk].sort()[0]

                fullstate_timeframes[chunk_idx] = random_init_timeframes + chunk_idx

            assert fullstate_timeframes.shape == (n_chunks, self.n_fullstate_timeframes_per_chunk)
            return fullstate_timeframes


class CFDDataset(Dataset, DatasetMixin):

    def __init__(
        self, 
        root: str, 
        init_sensor_timeframes: List[int],
        future_prediction_range: List[int] | None,
        n_fullstate_timeframes_per_chunk: int,
        n_samplings_per_chunk: int,
        resolution: Tuple[int, int],
        n_sensors: int,
        dropout_probabilities: List[float],
        noise_level: float,
        sensor_generator: Literal['LHS', 'AroundCylinder'], 
        embedding_generator: Literal['Voronoi', 'Mask', 'Vector'],
        init_fullstate_timeframes: List[int] | None,
        seed: int,
        write_to_disk: bool = True,
    ) -> None:
        
        super().__init__()
        # self.case_directories: List[str] = sorted([os.path.join(root, case_dir) for case_dir in os.listdir(root)])
        npy_files = [f for f in os.listdir(root) if f.endswith('.npy')]
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {root}")
        self.data_file = os.path.join(root, npy_files[0])
        self.case_name = os.path.basename(root)

        self.root: str = root
        self.init_sensor_timeframes: List[int] = init_sensor_timeframes
        self.future_prediction_range: List[int] | None = future_prediction_range
        self.n_fullstate_timeframes_per_chunk: int = n_fullstate_timeframes_per_chunk
        self.n_samplings_per_chunk: int = n_samplings_per_chunk
        self.resolution: Tuple[int, int] = resolution
        self.n_sensors: int = n_sensors
        self.dropout_probabilities: List[float] = dropout_probabilities
        self.noise_level: float = noise_level
        self.init_fullstate_timeframes: List[int] | None = init_fullstate_timeframes
        self.seed: int = seed
        self.is_random_fullstate_frames: bool = init_fullstate_timeframes is None
        self.write_to_disk: bool = write_to_disk

        self.H, self.W = resolution
        self.n_sensor_timeframes_per_chunk: int = len(init_sensor_timeframes)

        self.dest: str = os.path.join('tensors', os.path.basename(root))
        self.sensor_timeframes_dest: str = os.path.join(self.dest, 'sensor_timeframes')
        self.sensor_values_dest: str = os.path.join(self.dest, 'sensor_values')
        self.fullstate_timeframes_dest: str = os.path.join(self.dest, 'fullstate_timeframes')
        self.fullstate_values_dest: str = os.path.join(self.dest, 'fullstate_values')
        self.sensor_positions_dest: str = os.path.join(self.dest, 'sensor_positions')
        self.metadata_dest: str = os.path.join(self.dest, 'metadata')

        if not self.is_random_fullstate_frames:
            # NOTE: fullstate frames are deterministically generated
            if n_fullstate_timeframes_per_chunk != len(init_fullstate_timeframes):
                raise ValueError(
                    f'n_fullstate_timeframes_per_chunk should be logically set to len(init_fullstate_timeframes) when sensors are generated deterministically, '
                    f'get: n_fullstate_timeframes_per_chunk={n_fullstate_timeframes_per_chunk} and init_fullstate_timeframes={init_fullstate_timeframes}'
                )
            if n_samplings_per_chunk != 1:
                raise ValueError(
                    f'n_samplings_per_chunk should be logically set to 1 when sensors are generated deterministically, '
                    f'get: {n_samplings_per_chunk}'
                )

        if sensor_generator == 'LHS':
            self.sensor_generator = LHS(n_sensors=n_sensors)
            self.sensor_generator.seed = seed
            self.sensor_generator.resolution = resolution
            self.sensor_positions = self.sensor_generator()
        else:
            self.sensor_generator = AroundCylinder(n_sensors=n_sensors)
            self.sensor_generator.seed = seed
            self.sensor_generator.resolution = resolution
            self.sensor_positions = self.sensor_generator(
                hw_meters=(0.14, 0.24), center_hw_meters=(0.07, 0.065), radius_meters=0.03,
            )
        
        assert self.sensor_positions.shape == (self.sensor_generator.n_sensors, 2)
        if embedding_generator == 'Mask':
            self.embedding_generator = Mask(
                resolution=resolution, sensor_positions=self.sensor_positions, 
                dropout_probabilities=dropout_probabilities, noise_level=noise_level,
            )
        elif embedding_generator == 'Voronoi':
            self.embedding_generator = Voronoi(
                resolution=resolution, sensor_positions=self.sensor_positions, 
                dropout_probabilities=dropout_probabilities, noise_level=noise_level,
            )
        else:
            self.embedding_generator = Vector(
                resolution=resolution, sensor_positions=self.sensor_positions, 
                dropout_probabilities=dropout_probabilities, noise_level=noise_level,
            )

        self.case_names: List[str] = []
        self.sampling_ids: List[int] = []
        if self.write_to_disk:
            self.__write2disk()
        else:
            self.__load_metadata()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix: str = f'_{self.case_names[idx]}_{self.sampling_ids[idx]}_'
        suffix: str = str(idx).zfill(6)
        sensor_timeframe_tensor: torch.Tensor = torch.load(
            os.path.join(self.sensor_timeframes_dest, f'st{prefix}{suffix}.pt'), 
            weights_only=True
        )
        sensor_tensor: torch.Tensor = torch.load(
            os.path.join(self.sensor_values_dest, f'sv{prefix}{suffix}.pt'), 
            weights_only=True
        ).float()
        fullstate_timeframe_tensor: torch.Tensor = torch.load(
            os.path.join(self.fullstate_timeframes_dest, f'ft{prefix}{suffix}.pt'), 
            weights_only=True
        )
        fullstate_tensor: torch.Tensor = torch.load(
            os.path.join(self.fullstate_values_dest, f'fv{prefix}{suffix}.pt'), 
            weights_only=True
        ).float()
        case_name: str = self.case_names[idx]
        sampling_id: int = self.sampling_ids[idx]
        return sensor_timeframe_tensor, sensor_tensor, fullstate_timeframe_tensor, fullstate_tensor, case_name, sampling_id
    
    def __len__(self) -> int:
        return len([f for f in os.listdir(self.fullstate_values_dest) if f.endswith('.pt')])

    def __load_metadata(self) -> None:
        """
        Load metadata when skipping write-to-disk. Requires existing tensors directory.
        """
        metadata_file = os.path.join(self.metadata_dest, 'metadata.json')
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(
                f'Metadata not found at {metadata_file}. '
                'Either generate tensors once (write_to_disk=True) or ensure precomputed tensors exist.'
            )
        with open(metadata_file, 'r') as f:
            records: List[Dict[str, Any]] = json.load(f)
        for record in records:
            self.case_names.append(record['case_name'])
            self.sampling_ids.append(record['sampling_id'])

    def __write2disk(self) -> None:
        # prepare dest directories
        if os.path.isdir(self.dest): shutil.rmtree(self.dest)
        os.makedirs(name=self.sensor_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.sensor_values_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_timeframes_dest, exist_ok=True)
        os.makedirs(name=self.fullstate_values_dest, exist_ok=True)
        os.makedirs(name=self.sensor_positions_dest, exist_ok=True)
        os.makedirs(name=self.metadata_dest, exist_ok=True)
        
        # save position tensors for reference
        torch.save(obj=self.sensor_positions, f=os.path.join(self.sensor_positions_dest, 'pos.pt'))

        sensor_timeframes_list: List[List[int]] = []
        fullstate_timeframes_list: List[List[int]] = []
        running_index: int = 0

        data = torch.from_numpy(np.load(self.data_file)).cuda().float()
        # (B, H, W, C) -> (B, C, H, W)
        if data.ndim == 4:
            data = data.permute(0, 3, 2, 1)
        
        n_channels = data.shape[1]
        case_id = 0
        total_timeframes = data.shape[0]

        max_frame_idx = max(self.init_sensor_timeframes)
        if self.init_fullstate_timeframes is not None:
            max_frame_idx = max(max_frame_idx, max(self.init_fullstate_timeframes))
        if self.future_prediction_range is not None:
            max_frame_idx = max(max_frame_idx, max(self.init_sensor_timeframes) + self.future_prediction_range[1])

        n_chunks: int = total_timeframes - max_frame_idx
        if n_chunks <= 0:
            raise ValueError(
                f'Data has insufficient frames ({total_timeframes}) for the required range (max index: {max_frame_idx})'
            )
        sensor_timeframes: torch.Tensor = self.prepare_sensor_timeframes(n_chunks=n_chunks)
            
        # sensor data
        sensor_frame_data: torch.Tensor = data[sensor_timeframes]
        sensor_frame_data = F.interpolate(input=sensor_frame_data.flatten(0, 1), size=self.resolution, mode='bicubic')
        sensor_frame_data = sensor_frame_data.reshape(n_chunks, self.n_sensor_timeframes_per_chunk, n_channels, self.H, self.W)
        
        for sampling_id in range(self.n_samplings_per_chunk):
            # fullstate data
            if self.is_random_fullstate_frames:
                print('Randomly generating fullstate frames')
                fullstate_timeframes: torch.Tensor = self.prepare_fullstate_timeframes(
                    n_chunks=n_chunks,
                    seed=self.seed + case_id + sampling_id,
                )
            else:
                print('Deterministically generating fullstate frames')
                fullstate_timeframes: torch.Tensor = self.prepare_fullstate_timeframes(
                    n_chunks=n_chunks,
                    init_fullstate_timeframes=self.init_fullstate_timeframes
                )

            # Write each sample to disk
            for idx in tqdm(range(n_chunks), desc=f'Case {case_id + 1} | Sampling {sampling_id + 1}: '):
                # case name & sampling id
                self.case_names.append(self.case_name)
                self.sampling_ids.append(sampling_id)
                prefix: str = f'_{self.case_name}_{sampling_id}_'
                # indexes
                suffix = str(running_index).zfill(6)
                # save sensor timeframes, sensor value (dynamic to chunks, but constant to samplings)
                sensor_timeframes_list.append(sensor_timeframes[idx].tolist())
                torch.save(
                    obj=sensor_timeframes[idx].clone(),
                    f=os.path.join(self.sensor_timeframes_dest, f'st{prefix}{suffix}.pt')
                )
                sensor_sample: torch.Tensor = sensor_frame_data[idx:idx + 1]
                sensor_embedding = self.embedding_generator(
                    data=sensor_sample,
                    seed=self.seed + case_id + sampling_id + idx,
                ).squeeze(0)
                torch.save(obj=sensor_embedding.clone(), f=os.path.join(self.sensor_values_dest, f'sv{prefix}{suffix}.pt'))
                
                # save fullstate timeframes, fullstate data
                fullstate_timeframe_sample: torch.Tensor = fullstate_timeframes[idx]
                fullstate_timeframes_list.append(fullstate_timeframe_sample.tolist())
                torch.save(
                    obj=fullstate_timeframe_sample.clone(),
                    f=os.path.join(self.fullstate_timeframes_dest, f'ft{prefix}{suffix}.pt')
                )
                fullstate_sample: torch.Tensor = data[fullstate_timeframe_sample].unsqueeze(0)
                fullstate_sample = F.interpolate(
                    input=fullstate_sample.flatten(0, 1),
                    size=self.resolution,
                    mode='bicubic'
                )
                fullstate_sample = fullstate_sample.reshape(self.n_fullstate_timeframes_per_chunk, n_channels, self.H, self.W)
                torch.save(obj=fullstate_sample.clone(), f=os.path.join(self.fullstate_values_dest, f'fv{prefix}{suffix}.pt'))
                running_index += 1
            
            # manual garbage collection to optimize GPU RAM, otherwise likely lead to OutOfMemoryError
            torch.cuda.empty_cache()
        
        assert len(self.case_names) == len(self.sampling_ids) == len(sensor_timeframes_list) == len(fullstate_timeframes_list)
        records: List[Dict[str, Any]] = [
            {
                'case_name': case_name, 'sampling_id': sampling_id, 
                'sensor_timeframes': sensor_timeframes, 'fullstate_timeframes': fullstate_timeframes,
            }
            for case_name, sampling_id, sensor_timeframes, fullstate_timeframes in zip(
                self.case_names, self.sampling_ids, sensor_timeframes_list, fullstate_timeframes_list
            )
        ]
        with open(os.path.join(self.metadata_dest, 'metadata.json'), 'w') as f:
            json.dump(obj=records, fp=f, indent=2)
