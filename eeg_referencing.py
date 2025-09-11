#!/usr/bin/env python3
"""
EEG Paradox - EEG Re-referencing Module
Converts EEG data from Linked Ears to Average Reference using Cuban normative dataset

Based on:
- Pascual-Marqui et al. (1988) - Average Reference transformation
- Hernández et al. (1994) - Global Scale Factor normalization
- Cuban Normative Database integration

Copyright (C) 2025 EEG Paradox Clinical System Contributors
Licensed under GNU General Public License v3.0
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EEGReReferencing:
    """
    EEG Re-referencing Module for Linked Ears to Average Reference conversion
    
    Features:
    - Linked Ears to Average Reference transformation
    - Global Scale Factor (GSF) normalization
    - Integration with Cuban normative database
    - Cross-spectral matrix processing (19x19x49)
    - Z-score coherence comparison
    """
    
    def __init__(self, cuban_database=None):
        """Initialize the re-referencing module"""
        self.cuban_db = cuban_database
        
        # EEG electrode configuration (19-channel 10-20 system)
        self.electrodes = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'O2'
        ]
        
        # Frequency configuration (49 points from 0.39 to 19.11 Hz)
        self.frequencies = np.linspace(0.39, 19.11, 49)
        
        # Clinical frequency bands - ADJUSTED to match Cuban database range (0.39-19.11 Hz)
        self.frequency_bands = {
            'delta': (0.5, 3.5),           # Delta waves
            'theta': (4.0, 7.5),           # Theta waves
            'alpha': (8.0, 12.0),          # Alpha waves
            'smr': (12.0, 15.0),           # Sensory Motor Rhythm (SMR)
            'beta1': (15.0, 18.0),         # Low Beta
            'beta2': (18.0, 19.11),        # High Beta (limited by Cuban database)
            'beta': (12.0, 19.11)          # Combined Beta band (limited by Cuban database)
        }
        
        # Transformation matrix for Linked Ears to Average Reference
        self._setup_transformation_matrix()
        
    def _setup_transformation_matrix(self):
        """Setup the transformation matrix for Linked Ears to Average Reference"""
        n_channels = len(self.electrodes)
        
        # Create transformation matrix T
        # T = I - (1/N) * 1 * 1^T
        # where I is identity matrix, 1 is vector of ones, N is number of channels
        self.T = np.eye(n_channels) - (1.0 / n_channels) * np.ones((n_channels, n_channels))
        
        logger.info(f"✅ Transformation matrix created: {self.T.shape}")
        
    def load_cross_spectral_matrix(self, file_path: str, key: str = 'MCross') -> np.ndarray:
        """
        Load cross-spectral matrix from .mat file
        
        Args:
            file_path: Path to .mat file
            key: Key for the cross-spectral matrix (default: 'MCross')
            
        Returns:
            Cross-spectral matrix (19x19x49)
        """
        try:
            logger.info(f"📁 Loading cross-spectral matrix from: {file_path}")
            
            # Load .mat file
            mat_data = sio.loadmat(file_path)
            
            if key not in mat_data:
                available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                raise KeyError(f"Key '{key}' not found. Available keys: {available_keys}")
            
            mcross = mat_data[key]
            
            # Validate dimensions
            if mcross.ndim != 3:
                raise ValueError(f"Expected 3D array, got {mcross.ndim}D")
            
            if mcross.shape != (19, 19, 49):
                logger.warning(f"⚠️ Expected shape (19,19,49), got {mcross.shape}")
                # Try to reshape if possible
                if mcross.size == 19 * 19 * 49:
                    mcross = mcross.reshape(19, 19, 49)
                    logger.info("✅ Reshaped to (19,19,49)")
                else:
                    raise ValueError(f"Cannot reshape to (19,19,49), total elements: {mcross.size}")
            
            logger.info(f"✅ Loaded MCross: {mcross.shape}, dtype: {mcross.dtype}")
            return mcross
            
        except Exception as e:
            logger.error(f"❌ Error loading cross-spectral matrix: {e}")
            raise
    
    def linked_ears_to_average_reference(self, mcross: np.ndarray) -> np.ndarray:
        """
        Convert cross-spectral matrix from Linked Ears to Average Reference
        
        Args:
            mcross: Cross-spectral matrix (19x19x49) in Linked Ears reference
            
        Returns:
            Cross-spectral matrix (19x19x49) in Average Reference
        """
        try:
            logger.info("🔄 Converting Linked Ears to Average Reference...")
            
            if mcross.shape != (19, 19, 49):
                raise ValueError(f"Expected shape (19,19,49), got {mcross.shape}")
            
            # Apply transformation: MCross_AR = T * MCross * T^T
            # For each frequency bin
            mcross_ar = np.zeros_like(mcross)
            
            for f in range(mcross.shape[2]):
                # Get cross-spectral matrix for this frequency
                m_freq = mcross[:, :, f]
                
                # Apply transformation: T * M * T^T
                m_ar = self.T @ m_freq @ self.T.T
                
                # Store result
                mcross_ar[:, :, f] = m_ar
                
                # Log progress every 10 frequencies
                if f % 10 == 0:
                    logger.debug(f"   Processed frequency {f+1}/49")
            
            logger.info("✅ Conversion to Average Reference completed")
            return mcross_ar
            
        except Exception as e:
            logger.error(f"❌ Error in re-referencing: {e}")
            raise
    
    def apply_global_scale_factor(self, mcross_ar: np.ndarray, 
                                reference_power: Optional[float] = None) -> np.ndarray:
        """
        Apply Global Scale Factor (GSF) normalization
        
        Args:
            mcross_ar: Cross-spectral matrix in Average Reference
            reference_power: Reference power level (if None, uses log-power average)
            
        Returns:
            Normalized cross-spectral matrix
        """
        try:
            logger.info("⚖️ Applying Global Scale Factor normalization...")
            
            if mcross_ar.shape != (19, 19, 49):
                raise ValueError(f"Expected shape (19,19,49), got {mcross_ar.shape}")
            
            # Calculate diagonal elements (power spectrum)
            power_spectrum = np.diagonal(mcross_ar, axis1=0, axis2=1).T  # Shape: (49, 19)
            
            # Calculate log-power average across channels for each frequency
            log_power_avg = np.mean(np.log10(power_spectrum + 1e-10), axis=1)  # Shape: (49,)
            
            if reference_power is None:
                # Use median of log-power average as reference
                reference_power = np.median(log_power_avg)
                logger.info(f"   Using median log-power as reference: {reference_power:.3f}")
            else:
                logger.info(f"   Using provided reference power: {reference_power:.3f}")
            
            # Calculate GSF for each frequency
            gsf = reference_power - log_power_avg  # Shape: (49,)
            
            # Apply GSF normalization
            mcross_normalized = np.zeros_like(mcross_ar)
            for f in range(mcross_ar.shape[2]):
                mcross_normalized[:, :, f] = mcross_ar[:, :, f] * (10 ** gsf[f])
            
            logger.info("✅ GSF normalization completed")
            return mcross_normalized
            
        except Exception as e:
            logger.error(f"❌ Error in GSF normalization: {e}")
            raise
    
    def compute_coherence_z_scores(self, mcross_ar: np.ndarray, 
                                 patient_age: float, 
                                 patient_sex: str = 'unknown') -> Dict[str, np.ndarray]:
        """
        Compute Z-scores for coherence by comparing to Cuban normative database
        
        Args:
            mcross_ar: Cross-spectral matrix in Average Reference
            patient_age: Patient age
            patient_sex: Patient sex
            
        Returns:
            Dictionary containing Z-scores for each frequency band
        """
        try:
            if self.cuban_db is None:
                logger.warning("⚠️ Cuban database not available, skipping Z-score computation")
                return {}
            
            logger.info("🔍 Computing coherence Z-scores using Cuban normative database...")
            
            # Extract coherence values from cross-spectral matrix
            coherence_data = {}
            
            for band_name, (f_min, f_max) in self.frequency_bands.items():
                # Find frequency indices for this band
                freq_indices = np.where((self.frequencies >= f_min) & (self.frequencies <= f_max))[0]
                
                if len(freq_indices) == 0:
                    continue
                
                # Extract coherence for this band
                band_coherence = []
                for f_idx in freq_indices:
                    # Get cross-spectral matrix for this frequency
                    m_freq = mcross_ar[:, :, f_idx]
                    
                    # Calculate coherence: |C_ij| = |M_ij| / sqrt(M_ii * M_jj)
                    power = np.diagonal(m_freq)
                    coherence = np.abs(m_freq) / np.sqrt(np.outer(power, power) + 1e-10)
                    
                    # Get upper triangular elements (avoid duplicates)
                    upper_tri = np.triu_indices_from(coherence, k=1)
                    band_coherence.extend(coherence[upper_tri])
                
                if band_coherence:
                    coherence_data[band_name] = np.array(band_coherence)
                    logger.info(f"   {band_name}: {len(band_coherence)} coherence values")
            
            # Compute Z-scores using Cuban database
            z_scores = {}
            for band_name, coherence_values in coherence_data.items():
                try:
                    # Create patient data dictionary
                    patient_data = {f'coherence_{band_name}': coherence_values}
                    
                    # Get Z-scores from Cuban database
                    band_z_scores = self.cuban_db.compute_precise_z_scores(
                        patient_data, patient_age, patient_sex
                    )
                    
                    if band_z_scores:
                        z_scores[band_name] = band_z_scores
                        logger.info(f"   ✅ {band_name}: Z-score range [{np.min(list(band_z_scores.values())):.2f}, {np.max(list(band_z_scores.values())):.2f}]")
                    else:
                        logger.warning(f"   ⚠️ No Z-scores computed for {band_name}")
                        
                except Exception as e:
                    logger.warning(f"   ⚠️ Error computing Z-scores for {band_name}: {e}")
                    continue
            
            logger.info(f"✅ Computed Z-scores for {len(z_scores)} frequency bands")
            return z_scores
            
        except Exception as e:
            logger.error(f"❌ Error computing coherence Z-scores: {e}")
            return {}
    
    def save_cross_spectral_matrix(self, mcross: np.ndarray, 
                                 output_path: str, 
                                 key: str = 'MCross_AR') -> bool:
        """
        Save cross-spectral matrix to .mat file
        
        Args:
            mcross: Cross-spectral matrix to save
            output_path: Output file path
            key: Key for the saved matrix
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"💾 Saving cross-spectral matrix to: {output_path}")
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to .mat file
            sio.savemat(output_path, {key: mcross})
            
            logger.info(f"✅ Saved {key}: {mcross.shape} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving cross-spectral matrix: {e}")
            return False
    
    def process_eeg_file(self, input_path: str, 
                        output_path: str,
                        patient_age: float,
                        patient_sex: str = 'unknown',
                        apply_gsf: bool = True,
                        compute_z_scores: bool = True) -> Dict[str, Any]:
        """
        Complete EEG processing pipeline: Load -> Re-reference -> Normalize -> Save -> Z-scores
        
        Args:
            input_path: Input .mat file path
            output_path: Output .mat file path
            patient_age: Patient age
            patient_sex: Patient sex
            apply_gsf: Whether to apply Global Scale Factor normalization
            compute_z_scores: Whether to compute Z-scores
            
        Returns:
            Dictionary containing processing results and statistics
        """
        try:
            logger.info("🚀 Starting complete EEG processing pipeline...")
            
            results = {
                'input_path': input_path,
                'output_path': output_path,
                'patient_age': patient_age,
                'patient_sex': patient_sex,
                'processing_steps': [],
                'statistics': {},
                'z_scores': {},
                'success': False
            }
            
            # Step 1: Load cross-spectral matrix
            logger.info("📁 Step 1: Loading cross-spectral matrix...")
            mcross_le = self.load_cross_spectral_matrix(input_path)
            results['processing_steps'].append('load')
            results['statistics']['input_shape'] = mcross_le.shape
            results['statistics']['input_dtype'] = str(mcross_le.dtype)
            
            # Step 2: Convert to Average Reference
            logger.info("🔄 Step 2: Converting to Average Reference...")
            mcross_ar = self.linked_ears_to_average_reference(mcross_le)
            results['processing_steps'].append('re_reference')
            results['statistics']['ar_shape'] = mcross_ar.shape
            
            # Step 3: Apply GSF normalization (optional)
            if apply_gsf:
                logger.info("⚖️ Step 3: Applying Global Scale Factor normalization...")
                mcross_normalized = self.apply_global_scale_factor(mcross_ar)
                results['processing_steps'].append('gsf_normalization')
                results['statistics']['normalized_shape'] = mcross_normalized.shape
                final_matrix = mcross_normalized
            else:
                final_matrix = mcross_ar
            
            # Step 4: Save processed matrix
            logger.info("💾 Step 4: Saving processed matrix...")
            save_success = self.save_cross_spectral_matrix(final_matrix, output_path)
            if save_success:
                results['processing_steps'].append('save')
            else:
                raise RuntimeError("Failed to save processed matrix")
            
            # Step 5: Compute Z-scores (optional)
            if compute_z_scores and self.cuban_db is not None:
                logger.info("🔍 Step 5: Computing Z-scores...")
                z_scores = self.compute_coherence_z_scores(
                    final_matrix, patient_age, patient_sex
                )
                results['z_scores'] = z_scores
                results['processing_steps'].append('z_scores')
                
                # Add Z-score statistics
                for band_name, band_z_scores in z_scores.items():
                    if band_z_scores:
                        z_values = list(band_z_scores.values())
                        results['statistics'][f'{band_name}_z_score_range'] = [
                            float(np.min(z_values)), float(np.max(z_values))
                        ]
            
            results['success'] = True
            logger.info("🎉 EEG processing pipeline completed successfully!")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ EEG processing pipeline failed: {e}")
            results['error'] = str(e)
            return results

# Convenience functions
def convert_linked_ears_to_average_reference(input_path: str, 
                                           output_path: str,
                                           patient_age: float,
                                           patient_sex: str = 'unknown',
                                           cuban_database=None) -> bool:
    """
    Convenience function for quick conversion
    
    Args:
        input_path: Input .mat file path
        output_path: Output .mat file path
        patient_age: Patient age
        patient_sex: Patient sex
        cuban_database: Optional Cuban database instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        re_ref = EEGReReferencing(cuban_database)
        results = re_ref.process_eeg_file(
            input_path, output_path, patient_age, patient_sex
        )
        return results['success']
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        return False

if __name__ == "__main__":
    # Test the re-referencing module
    print("🧠⚡ EEG Paradox - EEG Re-referencing Module Test ⚡🧠")
    print("=" * 70)
    
    # Create test instance
    re_ref = EEGReReferencing()
    
    # Test transformation matrix
    print(f"✅ Transformation matrix shape: {re_ref.T.shape}")
    print(f"✅ Electrodes: {len(re_ref.electrodes)}")
    print(f"✅ Frequencies: {len(re_ref.frequencies)}")
    print(f"✅ Frequency bands: {len(re_ref.frequency_bands)}")
    
    print("\n🎉 Re-referencing module ready for use!")
    print("\n📋 Usage:")
    print("   # Quick conversion:")
    print("   success = convert_linked_ears_to_average_reference('input.mat', 'output.mat', 25, 'M')")
    print("\n   # Full pipeline:")
    print("   re_ref = EEGReReferencing(cuban_database)")
    print("   results = re_ref.process_eeg_file('input.mat', 'output.mat', 25, 'M')")
