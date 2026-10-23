# Sourced by the mixed-calibration RCO jobs: CALIB (relative to glm53 or absolute) -> CAL, NAME, PFX.
#   calib53/mix_v1 -> PFX mix  (rco_runs/mix_k144_v2, rco_runs/prep_mix_v1, rco_runs/reap_mix)
#   calib53/mix_v2 -> PFX mix2 (rco_runs/mix2_k144_v2, rco_runs/prep_mix_v2, rco_runs/reap_mix2)
P=/work/glm53
CALIB=${CALIB:-${CALIB_DIR:-calib53/mix_v1}}
case $CALIB in /*) CAL=$CALIB;; *) CAL=$P/$CALIB;; esac
NAME=$(basename $CAL)
if [ "$NAME" = mix_v1 ]; then PFX=mix; else PFX=${NAME/_v/}; fi
