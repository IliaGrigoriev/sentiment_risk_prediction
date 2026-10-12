import csv

import pandas as pd

from const import FLAT_CMPL_PATH


class ComplaintLoader:

    _COLUMNS = [
        "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT",
        "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC",
        "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES",
        "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN",
        "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN",
        "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED", "DOT",
        "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN",
        "MANUF_DT", "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME",
        "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP",
        "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN",
        "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
    ]

    # Load the NHTSA flat complaints file
    # -----------------------------------
    def load(self):
        return pd.read_csv(
            FLAT_CMPL_PATH,
            sep="\t",
            header=None,
            names=self._COLUMNS,
            dtype="string",
            quoting=csv.QUOTE_NONE,
        )
