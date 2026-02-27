# Production Run Status

## Command
```bash
tmux attach -t produce
# or check log:
tail -f /mnt/data/yurh/Infinigen-Sim/production_run.log
```

## Run Details
- **Started**: 2026-02-25
- **Command**: `python -u batch_produce.py --all --num_seeds 3`
- **Scope**: 81 factories x 3 seeds = 239 objects
- **Sources**: 64 IM + 13 PhysXNet + 4 PhysX_mobility

## Pipeline Phases
1. Phase 1 (Setup scenes): 51 PhysXNet/PhysXMob objects set up, 188 IM skipped (already exist)
2. Phase 2 (Precompute splits): In progress...
3. Phase 3 (Render videos): Pending

## Output Locations
- Precomputed splits: `precompute/{Factory}/{identifier}/part0.obj + part1.obj + metadata.json`
- Rendered videos: `outputs/motion_videos/{Factory}/{seed}/*.mp4`
- Log: `production_run.log`

## Known Issues
- BagPhysXNetFactory/12008: skipped (no movable joints)

## Quick Verification
```bash
# Count completed precompute dirs
find precompute/ -name metadata.json | wc -l

# Count rendered videos
find outputs/motion_videos/ -name "*.mp4" | wc -l

# Check for failures
grep "FAIL" production_run.log
```
