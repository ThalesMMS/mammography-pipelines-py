# apps

## Purpose
Umbrella package for operator-facing applications. These directories expose desktop or web
interfaces for labeling, annotation, browsing, and experiment monitoring on top of the same
underlying pipeline code used by the CLI.

## Entry Points and Key Modules
- There is no single launcher at this level; concrete entrypoints live in the child application
packages.
- CLI wrappers in `mammography.commands` expose these apps as subcommands such as `label-density`,
`label-patches`, and `web`.

### Subdirectories
- [`density_classifier/`](density_classifier/README.md): Manual breast-density labeling and review
UI.
- [`patch_marking/`](patch_marking/README.md): ROI annotation interface for patch-based workflows.
- [`web_ui/`](web_ui/README.md): Main Streamlit dashboard for interactive dataset browsing,
inference, explainability, experiment review, training configuration, and hyperparameter tuning.

## How It Fits into the Pipeline
- Provides human-in-the-loop tooling for data curation and experiment review.
- Keeps UI-specific state and presentation logic out of the lower-level data and training modules.
- Groups the desktop-style density and ROI tools separately from the broader Streamlit dashboard.

## Inputs and Outputs
- Inputs: DICOM studies, CSV labels, model checkpoints, experiment directories, and user
interactions from desktop or Streamlit sessions.
- Outputs: updated annotations, saved crops, reviewed labels, and interactive dashboards rather than
batch pipeline artifacts.

## Dependencies
- Internal: [`apps/density_classifier`](density_classifier/README.md),
[`apps/patch_marking`](patch_marking/README.md), [`apps/web_ui`](web_ui/README.md).
- External: UI libraries such as `streamlit`, `cv2`, and `matplotlib` depending on the child app.

## Extension and Maintenance Notes
- Treat this package as a container for applications, not as a dumping ground for generic helpers;
reusable logic should stay in lower-level packages and be imported here.
- If a new app shares data loading or visualization behavior with an existing one, extract that
logic into the corresponding core package instead of duplicating it.
- Keep app-specific state management local to each child package so different UIs can evolve
independently.

## Related Directories
- [`apps/density_classifier`](density_classifier/README.md): Manual breast-density labeling and
review UI.
- [`apps/patch_marking`](patch_marking/README.md): ROI annotation interface for patch-based
workflows.
- [`apps/web_ui`](web_ui/README.md): Main Streamlit dashboard for interactive dataset browsing,
inference, explainability, experiment review, training configuration, and hyperparameter tuning.
- [`commands`](../commands/README.md): Internal command handlers behind the top-level `mammography`
CLI.
