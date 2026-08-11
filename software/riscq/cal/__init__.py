"""riscq.cal — the calibration suite (spec 06): the fit helpers (`fits`), the YAML config tree
(`Config`), and the calibration classes (Amplitude, Frequency, Phase, T1, T2, ReadoutCalibration,
Separation, Fidelity, ReadoutFidelity, Window, Punchout, Resonator) plus the autocalibration
sequences."""

from riscq.cal import fits
from riscq.cal.base import Result
from riscq.cal.config import Config
from riscq.cal.drag import Leakage, optimize_fast_drag
from riscq.cal.qubit import Amplitude, EFAmplitude, EFFrequency, EFPhase, Frequency, Phase, T1, T2
from riscq.cal.readout import (Classifier, ClassifierN, Fidelity, Punchout, ReadoutCalibration,
                               ReadoutFidelity, Resonator, Separation, Window, rcorr)
from riscq.cal.rpe import (Angles, CZRPE, RPEAmplitude, RPEBranchError, RPEFrequency, RPEPhase,
                           cz_angles, damped_update, freq_error_hz, idle_angles, vz_correction,
                           x90_angles)
from riscq.cal.sequences import calibration_x6y3
from riscq.cal.twoqubit import (JAZZ, CZAmpFreqSweep, CZAmplitude, CZFrequency, CZSweep,
                                LocalPhases, RelativePhase, SpectatorPhase, calc_cz_frequency,
                                coupler_core, cz_coupler_form, cz_drive_table, cz_sandwich,
                                cz_table, joint_populations, pair_key)

__all__ = ["Config", "Result", "fits", "Amplitude", "Frequency", "Phase", "T1", "T2",
           "EFAmplitude", "EFFrequency", "EFPhase",
           "ReadoutCalibration", "ReadoutFidelity", "Separation", "Fidelity", "Window",
           "Punchout", "Resonator", "rcorr", "Leakage", "optimize_fast_drag",
           "Angles", "RPEBranchError", "RPEAmplitude", "RPEFrequency", "RPEPhase", "CZRPE",
           "cz_angles", "damped_update", "freq_error_hz", "idle_angles", "vz_correction",
           "x90_angles",
           "Classifier", "ClassifierN", "calibration_x6y3",
           "JAZZ", "CZSweep", "CZFrequency", "CZAmpFreqSweep", "CZAmplitude", "LocalPhases",
           "RelativePhase", "SpectatorPhase",
           "calc_cz_frequency", "coupler_core", "cz_coupler_form", "cz_drive_table", "cz_sandwich",
           "cz_table", "joint_populations", "pair_key"]
