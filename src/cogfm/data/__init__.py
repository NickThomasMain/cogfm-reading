"""Reading data: adapters, batching and cross-validation splits.

Axis convention for every modality signal
-----------------------------------------
A modality signal is stored as ``(time, features)``: the variable axis first,
the fixed one second. An eye-tracking scanpath is ``(n_fixations, 3)``, an EEG
sentence window is ``(n_timepoints, n_channels)``, a parcellated fMRI window is
``(n_timepoints, n_parcels)``, and an event-related fMRI pattern is a sequence
of length one.

Batching and masking depend on the variable axis being first, so one padding
routine serves all modalities. Encoders transpose internally where their
backbone expects channels first. Getting this wrong produces no error, only
silently transposed data, so adapters convert to this layout on read rather
than passing the source layout through.

Note that padding masks cover the time axis only. Channel heterogeneity between
corpora, such as differing EEG montages, is a separate concern handled by the
encoders.
"""
