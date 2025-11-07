"""
Energy estimation calculations for motion capture data.
Implements multiple methods for estimating metabolic energy expenditure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from scipy.signal import savgol_filter
import logging

class EnergyCalculator:
    """
    Calculate energy expenditure estimates from motion capture data.
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize energy calculator.
        
        Args:
            sampling_rate: Sampling rate of motion capture data in Hz
        """
        self.sampling_rate = sampling_rate
        self.body_segments = self._define_body_segments()
        self.segment_masses = self._define_segment_masses()
        
    def _define_body_segments(self) -> Dict[str, List[str]]:
        """Define body segments and their constituent markers."""
        return {
            'head': ['head'],
            'torso': ['shoulder_l', 'shoulder_r'],
            'upper_arm_r': ['shoulder_r', 'elbow_r'],
            'forearm_r': ['elbow_r', 'wrist_r'],
            'upper_arm_l': ['shoulder_l', 'elbow_l'],
            'forearm_l': ['elbow_l', 'wrist_l'],
            'thigh_r': ['hip_front_r', 'hip_back_r', 'knee_over_r', 'knee_under_r'],
            'shank_r': ['knee_over_r', 'knee_under_r', 'foot_front_r', 'foot_back_r'],
            'foot_r': ['foot_front_r', 'foot_back_r'],
            'thigh_l': ['hip_front_l', 'hip_back_l', 'knee_over_l', 'knee_under_l'],
            'shank_l': ['knee_over_l', 'knee_under_l', 'foot_front_l', 'foot_back_l'],
            'foot_l': ['foot_front_l', 'foot_back_l']
        }
    
    def _define_segment_masses(self) -> Dict[str, float]:
        """Define relative masses of body segments (% of total body mass)."""
        return {
            'head': 0.081,
            'torso': 0.497,
            'upper_arm_r': 0.028,
            'forearm_r': 0.016,
            'upper_arm_l': 0.028,
            'forearm_l': 0.016,
            'thigh_r': 0.100,
            'shank_r': 0.0465,
            'foot_r': 0.0145,
            'thigh_l': 0.100,
            'shank_l': 0.0465,
            'foot_l': 0.0145
        }
    
    def smooth_data(self, data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
        """
        Smooth motion data using Savitzky-Golay filter.
        
        Args:
            data: Input data array
            window_length: Length of smoothing window
            polyorder: Order of polynomial for smoothing
            
        Returns:
            Smoothed data array
        """
        if len(data) < window_length:
            return data
        
        if data.ndim == 1:
            return savgol_filter(data, window_length, polyorder)
        else:
            return np.apply_along_axis(
                lambda x: savgol_filter(x, window_length, polyorder), 
                axis=0, arr=data
            )
    
    def calculate_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate velocity from position data.
        
        Args:
            positions: Position array of shape (n_frames, 3)
            
        Returns:
            Velocity array of shape (n_frames-1, 3)
        """
        dt = 1.0 / self.sampling_rate
        return np.diff(positions, axis=0) / dt
    
    def calculate_acceleration(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate acceleration from position data.
        
        Args:
            positions: Position array of shape (n_frames, 3)
            
        Returns:
            Acceleration array of shape (n_frames-2, 3)
        """
        velocities = self.calculate_velocity(positions)
        dt = 1.0 / self.sampling_rate
        return np.diff(velocities, axis=0) / dt
    
    def calculate_segment_com(self, df: pd.DataFrame, segment: str) -> np.ndarray:
        """
        Calculate center of mass for a body segment.
        
        Args:
            df: DataFrame with marker data
            segment: Name of body segment
            
        Returns:
            Array of shape (n_frames, 3) with segment CoM coordinates
        """
        if segment not in self.body_segments:
            raise ValueError(f"Unknown segment: {segment}")
        
        markers = self.body_segments[segment]
        segment_positions = []
        
        for marker in markers:
            try:
                x = df[f'{marker}.X'].values
                y = df[f'{marker}.Y'].values
                z = df[f'{marker}.Z'].values
                segment_positions.append(np.column_stack([x, y, z]))
            except KeyError:
                logging.warning(f"Marker {marker} not found for segment {segment}")
                continue
        
        if not segment_positions:
            return np.array([])
        
        # Calculate average position (simple CoM approximation) using nanmean
        return np.nanmean(segment_positions, axis=0)
    
    def calculate_whole_body_com(self, df: pd.DataFrame, body_mass: float = 70.0) -> np.ndarray:
        """
        Calculate whole-body center of mass.
        
        Args:
            df: DataFrame with marker data
            body_mass: Total body mass in kg
            
        Returns:
            Array of shape (n_frames, 3) with whole-body CoM coordinates
        """
        com_positions = []
        total_mass = 0
        
        for segment, mass_fraction in self.segment_masses.items():
            segment_com = self.calculate_segment_com(df, segment)
            if segment_com.size > 0:
                segment_mass = body_mass * mass_fraction
                com_positions.append(segment_com * segment_mass)
                total_mass += segment_mass
        
        if not com_positions:
            return np.array([])
        
        # Weighted average of segment CoMs using nansum
        return np.nansum(com_positions, axis=0) / total_mass
    
    def calculate_mechanical_energy(self, df: pd.DataFrame, body_mass: float = 70.0) -> Dict[str, np.ndarray]:
        """
        Calculate mechanical energy components.
        
        Args:
            df: DataFrame with marker data
            body_mass: Total body mass in kg
            
        Returns:
            Dictionary with kinetic, potential, and total mechanical energy
        """
        com = self.calculate_whole_body_com(df, body_mass)
        if com.size == 0:
            return {}
        
        # Convert from mm to m if needed (detect if values are too large)
        if np.nanmax(np.abs(com)) > 10:  # Likely in mm
            com = com / 1000.0
        
        # Smooth CoM data
        com_smooth = self.smooth_data(com)
        
        # Calculate velocities
        velocities = self.calculate_velocity(com_smooth)
        speed = np.sqrt(np.nansum(velocities**2, axis=1))  # Use nansum for speed calculation
        
        # Kinetic energy: KE = 0.5 * m * v^2 (now in Joules)
        kinetic_energy = 0.5 * body_mass * speed**2
        
        # Potential energy: PE = m * g * h (using Y as vertical)
        g = 9.81  # gravity
        # Use relative height (subtract minimum height to avoid huge PE values)
        min_height = np.nanmin(com_smooth[:, 1])
        relative_height = com_smooth[:-1, 1] - min_height
        potential_energy = body_mass * g * relative_height
        
        # Total mechanical energy
        total_energy = kinetic_energy + potential_energy
        
        return {
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': total_energy,
            'com_positions': com_smooth,
            'velocities': velocities,
            'speeds': speed
        }
    
    def calculate_metabolic_energy_brockway(self, mechanical_energy: Dict[str, np.ndarray], 
                                          efficiency: float = 0.25) -> np.ndarray:
        """
        Estimate metabolic energy using mechanical work and efficiency.
        
        Args:
            mechanical_energy: Dictionary from calculate_mechanical_energy
            efficiency: Mechanical efficiency (typically 0.20-0.30)
            
        Returns:
            Estimated metabolic energy expenditure rate (W)
        """
        if 'total' not in mechanical_energy:
            return np.array([])
        
        # Calculate rate of change of mechanical energy (mechanical power)
        total_energy = mechanical_energy['total']
        dt = 1.0 / self.sampling_rate
        mechanical_power = np.abs(np.diff(total_energy)) / dt
        
        # Estimate metabolic power (assuming some efficiency)
        metabolic_power = mechanical_power / efficiency
        
        return mechanical_power, metabolic_power
    
    def calculate_metabolic_energy_pandolf(self, df: pd.DataFrame, 
                                         body_mass: float = 70.0,
                                         walking_speed: Optional[float] = None) -> float:
        """
        Estimate metabolic energy using Pandolf equation for walking.
        NOTE: This method is designed for adult walking and may not be appropriate 
        for children's hopscotch activities.
        
        Args:
            df: DataFrame with marker data
            body_mass: Body mass in kg
            walking_speed: Walking speed in m/s (calculated if None)
            
        Returns:
            Estimated metabolic rate (W)
        """
        if walking_speed is None:
            # Calculate walking speed from CoM movement
            com = self.calculate_whole_body_com(df, body_mass)
            if com.size == 0:
                return 0.0
            
            velocities = self.calculate_velocity(com)
            # Use horizontal velocity components only
            horizontal_velocities = velocities[:, [0, 2]]  # X and Z
            horizontal_speeds = np.sqrt(np.nansum(horizontal_velocities**2, axis=1))
            walking_speed = np.nanmean(horizontal_speeds)
        
        # Convert position units from mm to m if needed (common in mocap data)
        if walking_speed > 10:  # Likely in mm/s, convert to m/s
            walking_speed = walking_speed / 1000.0
        
        # Pandolf equation for level walking (simplified)
        # VO2 = 3.5 + 0.2*speed + 0.9*speed*grade
        # For level walking, grade = 0
        speed_kmh = walking_speed * 3.6  # convert m/s to km/h
        vo2_ml_kg_min = 3.5 + 0.2 * speed_kmh
        
        # Convert VO2 to watts: 1 ml O2/kg/min = 20.1 J/min/kg = 0.335 W/kg
        # More accurate conversion: 1 L O2 ≈ 5 kcal = 20.93 kJ
        metabolic_rate = vo2_ml_kg_min * body_mass * (20.93 / 1000) / 60  # W
        
        return metabolic_rate
    
    def calculate_com_vertical_energy(self, df: pd.DataFrame, body_mass: float = 70.0) -> Dict[str, np.ndarray]:
        """
        Calculate energy expenditure based on center of mass vertical displacement.
        Appropriate for activities with significant vertical movement like hopscotch.
        
        Args:
            df: DataFrame with marker data
            body_mass: Body mass in kg
            
        Returns:
            Dictionary with vertical energy metrics
        """
        com = self.calculate_whole_body_com(df, body_mass)
        if com.size == 0:
            return {}
        
        # Convert from mm to m if needed (detect if values are too large)
        if np.nanmax(np.abs(com)) > 10:  # Likely in mm
            com = com / 1000.0
        
        # Smooth CoM data
        com_smooth = self.smooth_data(com)
        
        # Extract vertical (Y) component
        vertical_pos = com_smooth[:, 1]
        
        # Calculate vertical velocity and acceleration
        vertical_vel = self.calculate_velocity(vertical_pos.reshape(-1, 1)).flatten()
        vertical_acc = self.calculate_acceleration(vertical_pos.reshape(-1, 1)).flatten()
        
        # Gravity constant
        g = 9.81
        
        # Vertical kinetic energy: KE_v = 0.5 * m * v_y^2 (now in Joules)
        vertical_kinetic = 0.5 * body_mass * vertical_vel**2
        
        # Vertical potential energy relative to minimum height
        min_height = np.nanmin(vertical_pos)
        vertical_potential = body_mass * g * (vertical_pos[:-1] - min_height)
        
        # Total vertical mechanical energy
        total_vertical_energy = vertical_kinetic + vertical_potential
        
        # Rate of vertical energy change (power) - now in Watts
        dt = 1.0 / self.sampling_rate
        vertical_power = np.abs(np.diff(total_vertical_energy)) / dt
        
        # Vertical work done against gravity (positive work only) - now in Watts
        positive_work_rate = np.maximum(0, body_mass * g * vertical_vel)
        
        return {
            'vertical_position': vertical_pos,
            'vertical_velocity': vertical_vel,
            'vertical_acceleration': vertical_acc,
            'vertical_kinetic_energy': vertical_kinetic,
            'vertical_potential_energy': vertical_potential,
            'total_vertical_energy': total_vertical_energy,
            'vertical_power': vertical_power,
            'positive_work_rate': positive_work_rate
        }
    
    def calculate_com_velocity_variability(self, df: pd.DataFrame, body_mass: float = 70.0) -> Dict[str, float]:
        """
        Calculate energy metrics based on velocity variation of center of mass.
        Higher velocity variability indicates more energy expenditure.
        
        Args:
            df: DataFrame with marker data
            body_mass: Body mass in kg
            
        Returns:
            Dictionary with velocity variability metrics
        """
        com = self.calculate_whole_body_com(df, body_mass)
        if com.size == 0:
            return {}
        
        # Smooth CoM data
        com_smooth = self.smooth_data(com)
        
        # Calculate 3D velocities
        velocities = self.calculate_velocity(com_smooth)
        speeds = np.sqrt(np.nansum(velocities**2, axis=1))
        
        # Calculate accelerations (change in velocity)
        accelerations = self.calculate_acceleration(com_smooth)
        acceleration_magnitudes = np.sqrt(np.nansum(accelerations**2, axis=1))
        
        # Velocity variability metrics
        speed_std = np.nanstd(speeds)
        speed_cv = speed_std / np.nanmean(speeds) if np.nanmean(speeds) > 0 else 0
        
        # Acceleration-based metrics
        mean_acceleration = np.nanmean(acceleration_magnitudes)
        max_acceleration = np.nanmax(acceleration_magnitudes)
        
        # Jerk (rate of change of acceleration) 
        jerks = np.diff(acceleration_magnitudes) * self.sampling_rate
        mean_jerk = np.nanmean(np.abs(jerks))
        
        # Energy cost based on velocity variations
        # Higher variability = more energy for acceleration/deceleration
        variability_cost = speed_std * body_mass  # Simplified metric
        acceleration_cost = mean_acceleration * body_mass
        
        return {
            'speed_std': speed_std,
            'speed_cv': speed_cv,
            'mean_acceleration': mean_acceleration,
            'max_acceleration': max_acceleration,
            'mean_jerk': mean_jerk,
            'variability_energy_cost': variability_cost,
            'acceleration_energy_cost': acceleration_cost
        }
    
    def calculate_segmental_energy(self, df: pd.DataFrame, body_mass: float = 70.0) -> Dict[str, Any]:
        """
        Calculate energy expenditure based on segmental motion analysis.
        Considers movement of individual body segments.
        
        Args:
            df: DataFrame with marker data
            body_mass: Body mass in kg
            
        Returns:
            Dictionary with segmental energy metrics
        """
        segment_energies = {}
        total_kinetic_energy = []
        
        for segment, mass_fraction in self.segment_masses.items():
            segment_com = self.calculate_segment_com(df, segment)
            if segment_com.size == 0:
                continue
            
            # Smooth segment CoM
            segment_com_smooth = self.smooth_data(segment_com)
            
            # Calculate segment velocities
            segment_velocities = self.calculate_velocity(segment_com_smooth)
            segment_speeds = np.sqrt(np.nansum(segment_velocities**2, axis=1))
            
            # Segment mass
            segment_mass = body_mass * mass_fraction
            
            # Segment kinetic energy
            segment_ke = 0.5 * segment_mass * segment_speeds**2
            
            # Store segment data
            segment_energies[segment] = {
                'mass': segment_mass,
                'speeds': segment_speeds,
                'kinetic_energy': segment_ke,
                'mean_kinetic_energy': np.nanmean(segment_ke),
                'max_speed': np.nanmax(segment_speeds)
            }
            
            # Accumulate for total
            if len(total_kinetic_energy) == 0:
                total_kinetic_energy = segment_ke.copy()
            else:
                # Ensure arrays are same length
                min_len = min(len(total_kinetic_energy), len(segment_ke))
                total_kinetic_energy = total_kinetic_energy[:min_len] + segment_ke[:min_len]
        
        # Calculate segment coordination metrics
        # Measure synchrony between segments (lower values = better coordination)
        segment_speeds_matrix = []
        for segment in segment_energies:
            speeds = segment_energies[segment]['speeds']
            if len(speeds) > 0:
                segment_speeds_matrix.append(speeds)
        
        coordination_index = 0.0
        if len(segment_speeds_matrix) > 1:
            # Calculate coefficient of variation across segments at each time point
            speeds_array = np.array(segment_speeds_matrix)
            min_len = min(len(speeds) for speeds in segment_speeds_matrix)
            speeds_array = np.array([speeds[:min_len] for speeds in segment_speeds_matrix])
            
            # Coordination as inverse of speed variability across segments
            segment_cv_time = np.nanstd(speeds_array, axis=0) / (np.nanmean(speeds_array, axis=0) + 1e-6)
            coordination_index = np.nanmean(segment_cv_time)
        
        return {
            'segment_energies': segment_energies,
            'total_segmental_ke': total_kinetic_energy,
            'mean_total_segmental_ke': np.nanmean(total_kinetic_energy) if len(total_kinetic_energy) > 0 else 0.0,
            'coordination_index': coordination_index,
            'num_active_segments': len(segment_energies)
        }
    
    def calculate_energy_summary(self, df: pd.DataFrame, 
                               body_mass: float = 27.0,
                               trial_duration: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive energy summary for a trial including hopscotch-specific metrics.
        
        Args:
            df: DataFrame with marker data
            body_mass: Body mass in kg
            trial_duration: Trial duration in seconds (calculated if None)
            
        Returns:
            Dictionary with energy metrics
        """
        if trial_duration is None:
            trial_duration = len(df) / self.sampling_rate
        
        # Calculate traditional mechanical energy
        mech_energy = self.calculate_mechanical_energy(df, body_mass)
        
        if not mech_energy:
            return {}
        
        # Calculate metabolic estimates (traditional methods)
        mech_power, metab_power = self.calculate_metabolic_energy_brockway(mech_energy)
        pandolf_rate = self.calculate_metabolic_energy_pandolf(df, body_mass)
        
        # Calculate hopscotch-specific metrics
        vertical_energy = self.calculate_com_vertical_energy(df, body_mass)
        velocity_variability = self.calculate_com_velocity_variability(df, body_mass)
        segmental_energy = self.calculate_segmental_energy(df, body_mass)
        
        # Start with traditional summary statistics
        summary = {
            'trial_duration': trial_duration,
            'mean_speed': np.mean(mech_energy['speeds']) if 'speeds' in mech_energy else 0.0,
            'max_speed': np.max(mech_energy['speeds']) if 'speeds' in mech_energy else 0.0,
            'mean_kinetic_energy': np.mean(mech_energy['kinetic']) if 'kinetic' in mech_energy else 0.0,
            'mean_potential_energy': np.mean(mech_energy['potential']) if 'potential' in mech_energy else 0.0,
            'mean_total_energy': np.mean(mech_energy['total']) if 'total' in mech_energy else 0.0,
            'mean_mechanical_power': np.mean(mech_power) if len(mech_power) > 0 else 0.0,
            'mean_metabolic_power_efficiency': np.mean(metab_power) if len(metab_power) > 0 else 0.0,
            'metabolic_rate_pandolf': pandolf_rate,
            'total_mechanical_work': np.sum(mech_power) * (1.0/self.sampling_rate) if len(mech_power) > 0 else 0.0,
            'total_metabolic_energy_efficiency': np.sum(metab_power) * (1.0/self.sampling_rate) if len(metab_power) > 0 else 0.0,
            'total_metabolic_energy_pandolf': pandolf_rate * trial_duration
        }
        
        # Add vertical energy metrics (hopscotch-specific)
        if vertical_energy:
            summary.update({
                'vertical_displacement_range': np.nanmax(vertical_energy['vertical_position']) - np.nanmin(vertical_energy['vertical_position']) if 'vertical_position' in vertical_energy else 0.0,
                'mean_vertical_velocity': np.nanmean(np.abs(vertical_energy['vertical_velocity'])) if 'vertical_velocity' in vertical_energy else 0.0,
                'max_vertical_velocity': np.nanmax(np.abs(vertical_energy['vertical_velocity'])) if 'vertical_velocity' in vertical_energy else 0.0,
                'mean_vertical_acceleration': np.nanmean(np.abs(vertical_energy['vertical_acceleration'])) if 'vertical_acceleration' in vertical_energy else 0.0,
                'mean_vertical_kinetic_energy': np.nanmean(vertical_energy['vertical_kinetic_energy']) if 'vertical_kinetic_energy' in vertical_energy else 0.0,
                'mean_vertical_potential_energy': np.nanmean(vertical_energy['vertical_potential_energy']) if 'vertical_potential_energy' in vertical_energy else 0.0,
                'mean_vertical_power': np.nanmean(vertical_energy['vertical_power']) if 'vertical_power' in vertical_energy else 0.0,
                'mean_positive_work_rate': np.nanmean(vertical_energy['positive_work_rate']) if 'positive_work_rate' in vertical_energy else 0.0,
                'total_vertical_work': np.nansum(vertical_energy['positive_work_rate']) * (1.0/self.sampling_rate) if 'positive_work_rate' in vertical_energy else 0.0
            })
        
        # Add velocity variability metrics (hopscotch-specific)
        if velocity_variability:
            summary.update({
                'speed_variability_std': velocity_variability.get('speed_std', 0.0),
                'speed_variability_cv': velocity_variability.get('speed_cv', 0.0),
                'mean_acceleration_magnitude': velocity_variability.get('mean_acceleration', 0.0),
                'max_acceleration_magnitude': velocity_variability.get('max_acceleration', 0.0),
                'mean_jerk': velocity_variability.get('mean_jerk', 0.0),
                'variability_energy_cost': velocity_variability.get('variability_energy_cost', 0.0),
                'acceleration_energy_cost': velocity_variability.get('acceleration_energy_cost', 0.0)
            })
        
        # Add segmental energy metrics (hopscotch-specific)
        if segmental_energy:
            summary.update({
                'mean_segmental_kinetic_energy': segmental_energy.get('mean_total_segmental_ke', 0.0),
                'coordination_index': segmental_energy.get('coordination_index', 0.0),
                'active_segments_count': segmental_energy.get('num_active_segments', 0)
            })
            
            # Add per-segment energy summaries for key segments
            segment_energies = segmental_energy.get('segment_energies', {})
            for segment in ['thigh_r', 'thigh_l', 'shank_r', 'shank_l']:
                if segment in segment_energies:
                    summary[f'{segment}_mean_ke'] = segment_energies[segment].get('mean_kinetic_energy', 0.0)
                    summary[f'{segment}_max_speed'] = segment_energies[segment].get('max_speed', 0.0)
        
        # Calculate composite hopscotch energy metrics
        # These combine multiple aspects relevant to hopscotch movement
        composite_hopscotch_power = 0.0
        if vertical_energy and velocity_variability:
            # Combine power-based metrics (all in Watts) for hopscotch-specific activities
            vertical_power_component = summary.get('mean_vertical_power', 0.0)
            variability_power_component = summary.get('variability_energy_cost', 0.0) / trial_duration if trial_duration > 0 else 0.0
            acceleration_power_component = summary.get('acceleration_energy_cost', 0.0) / trial_duration if trial_duration > 0 else 0.0
            
            # Weight components based on importance for hopscotch movement
            composite_hopscotch_power = (
                0.6 * vertical_power_component +      # 60% vertical movement
                0.3 * variability_power_component +   # 30% movement variability  
                0.1 * acceleration_power_component    # 10% acceleration changes
            )
        
        summary['composite_hopscotch_power'] = composite_hopscotch_power
        summary['composite_hopscotch_energy'] = composite_hopscotch_power * trial_duration
        
        return summary
