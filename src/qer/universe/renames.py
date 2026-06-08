"""
Ticker rename and exit maps for S&P 500 membership reconstruction.

TICKER_RENAMES: same entity continued in the index under a new ticker symbol.
  Used at build time to collapse old symbols onto the current canonical one
  so historical membership intervals are attributed correctly (e.g. FB's
  pre-2022 membership rolls up under META).

TICKER_EXITS: entity ceased to exist (acquired/delisted) during the
  changes-log coverage but the log didn't record the removal event.
  Provides an explicit end date so membership closes at the right time
  rather than at the conservative log_end_date fallback.

Maintenance: when new corporate events occur, classify them and add to the
  appropriate map. See docs/data_limitations.md for the accuracy notes.
"""

# ---------------------------------------------------------------------------
# Ticker rename map.
#
# Maps OLD_TICKER -> CURRENT_TICKER. The build step walks the chain so
# multi-step renames (WLP -> ANTM -> ELV) collapse to the final symbol.
#
# Use ONLY for renames where the same entity continued in the index under
# a new ticker. For acquisitions (entity ceases to exist), use TICKER_EXITS.
# ---------------------------------------------------------------------------
TICKER_RENAMES = {
    "FB": "META",  # Facebook -> Meta Platforms (2022-06-09)
    "PCLN": "BKNG",  # Priceline -> Booking Holdings (2018-02-27)
    "KORS": "CPRI",  # Michael Kors -> Capri Holdings (2018-12-31)
    "COG": "CTRA",  # Cabot Oil & Gas -> Coterra (merger w/ Cimarex 2021-10-01)
    "WLTW": "WTW",  # Willis Towers Watson re-ticker (2022-01-11)
    "JEC": "J",  # Jacobs Engineering re-ticker (2022-01-25)
    "HRS": "LHX",  # Harris -> L3Harris (merger 2019-06-29)
    "LLL": "LHX",  # L3 Technologies -> L3Harris (same merger)
    "DLPH": "APTV",  # Delphi -> Aptiv (reorg 2017-12-04)
    "CDAY": "DAY",  # Ceridian -> Dayforce (2024-02-12)
    "FLT": "CPAY",  # Fleetcor -> Corpay (2024-03-25)
    "LUK": "JEF",  # Leucadia -> Jefferies Financial (2018-02-26)
    "RE": "EG",  # Everest Re re-ticker (2023-11-21)
    "TSO": "ANDV",  # Tesoro -> Andeavor (2017-08-01)
    # Note: ANDV was then acquired by Marathon Petroleum (MPC) on
    # 2018-10-01. ANDV stays as its own ticker with a closed interval
    # via TICKER_EXITS - the acquisition was a genuine exit, not a rename.
    "WLP": "ELV",  # WellPoint -> Anthem (2014) -> Elevance Health (2022)
    "ANTM": "ELV",  # Anthem -> Elevance Health (2022-06-28)
    "BBT": "TFC",  # BB&T -> Truist (merger w/ SunTrust 2019-12-09)
    "STI": "TFC",  # SunTrust -> Truist (same merger)
    "RTN": "RTX",  # Raytheon -> RTX (merger w/ United Technologies 2020-04-03)
    "UTX": "RTX",  # United Technologies -> RTX (same merger)
    "FISV": "FI",  # Fiserv re-ticker (2024-07-09)
    "FBHS": "FBIN",  # Fortune Brands Home & Security -> Fortune Brands Innovations (Dec 2022)
    "FII": "FHI",  # Federated Investors -> Federated Hermes (rebrand)
    "VIAB": "PARA",  # ViacomCBS class B -> Paramount (Feb 2022)
    "DISCA": "WBD",  # Discovery merged with WarnerMedia -> WBD (Apr 2022)
    "DISCK": "WBD",  # Discovery class K -> WBD (same merger)
    "FTR": "FYBR",  # Frontier Communications post-bankruptcy re-ticker (Apr 2021)
    "HFC": "DINO",  # HollyFrontier merger -> HF Sinclair / DINO (2022)
    "DISH": "SATS",  # DISH Network merged with EchoStar -> SATS (Dec 2023)
    "ARNC": "HWM",  # Arconic spinoff: HWM (Howmet) is the S&P 500 continuant
    "KFT": "MDLZ",  # Kraft Foods split: Mondelez kept the international snack business
    "DPS": "KDP",  # Dr Pepper Snapple merged with Keurig Green Mountain -> KDP (2018-07-09)
    "JNS": "JHG",  # Janus Capital merged with Henderson -> Janus Henderson (2017-05-30)
    "ETFC": "MS",  # E*TRADE acquired by Morgan Stanley (2020-10-02)
    # Note: ETFC -> MS is technically an acquisition by an existing index
    # member, so could go in TICKER_EXITS instead. Listed here because MS
    # was the operating entity continuation. Either classification is
    # defensible; pick one and stay consistent.
}


# ---------------------------------------------------------------------------
# Ticker exits map.
#
# For tickers ACQUIRED or DELISTED during the changes-log coverage where
# the log itself didn't record the removal event. Provides an explicit
# end date so the historical membership is preserved with correct exit
# timing rather than closed conservatively at log_end_date.
#
# Dates are accurate to within ~2 weeks based on public-source spot checks.
# Greater precision would require integrating S&P's official index-change
# announcements; see docs/data_limitations.md.
# ---------------------------------------------------------------------------
TICKER_EXITS = {
    # --- 2025 ---
    "ANSS": "2025-07-17",  # Ansys acquired by Synopsys
    "HES": "2025-07-18",  # Hess acquired by Chevron
    "WBA": "2025-08-28",  # Walgreens Boots Alliance taken private by Sycamore Partners
    # --- 2024 ---
    "CTLT": "2024-12-18",  # Catalent acquired by Novo Holdings
    "MRO": "2024-11-22",  # Marathon Oil acquired by ConocoPhillips
    "PXD": "2024-05-03",  # Pioneer Natural Resources acquired by Exxon
    "FSR": "2024-06-24",  # Fisker Inc. bankruptcy
    # --- 2023 ---
    "ATVI": "2023-10-13",  # Activision Blizzard acquired by Microsoft
    "FRC": "2023-05-01",  # First Republic seized by FDIC, sold to JPM
    "SIVB": "2023-03-10",  # Silicon Valley Bank failure
    "ENDP": "2023-04-27",  # Endo International bankruptcy/restructuring
    # --- 2022 ---
    "ABMD": "2022-12-22",  # Abiomed acquired by JNJ
    "TWTR": "2022-10-27",  # Twitter taken private by Musk
    "NLSN": "2022-10-11",  # Nielsen taken private
    "CERN": "2022-06-08",  # Cerner acquired by Oracle
    "XLNX": "2022-02-14",  # Xilinx acquired by AMD
    "DRE": "2022-10-03",  # Duke Realty acquired by Prologis
    "PBCT": "2022-04-01",  # People's United Bancorp acquired by M&T
    # --- 2021 ---
    "VAR": "2021-04-15",  # Varian acquired by Siemens Healthineers
    "MXIM": "2021-08-26",  # Maxim Integrated acquired by Analog Devices
    "TIF": "2021-01-07",  # Tiffany acquired by LVMH
    "KSU": "2021-12-14",  # Kansas City Southern acquired by Canadian Pacific
    "CXO": "2021-01-15",  # Concho Resources acquired by ConocoPhillips
    # --- 2020 ---
    "NBL": "2020-10-05",  # Noble Energy acquired by Chevron
    "WCG": "2020-01-23",  # WellCare acquired by Centene
    "AKS": "2020-03-13",  # AK Steel acquired by Cleveland-Cliffs
    "JCP": "2020-12-09",  # JCPenney bankruptcy
    "WIN": "2019-08-22",  # Windstream bankruptcy (listed under 2020 wave - check exact date)
    # --- 2019 ---
    "RHT": "2019-07-09",  # Red Hat acquired by IBM
    "DWDP": "2019-04-01",  # DowDuPont split into DD/DOW/CTVA
    "ESV": "2019-04-15",  # Ensco + Rowan -> Valaris
    # --- 2018 ---
    "MON": "2018-06-07",  # Monsanto acquired by Bayer
    "WYN": "2018-05-31",  # Wyndham split into WH and WYND
    # --- 2017 ---
    "JOYG": "2017-04-06",  # Joy Global acquired by Komatsu (taken private)
    "WFM": "2017-08-28",  # Whole Foods acquired by Amazon
    "STJ": "2017-01-04",  # St. Jude Medical acquired by Abbott
    # --- 2016 ---
    "SNDK": "2016-05-12",  # SanDisk acquired by Western Digital
    "CAM": "2016-04-06",  # Cameron International acquired by Schlumberger
    "PCP": "2016-01-29",  # Precision Castparts acquired by Berkshire Hathaway
    "TWC": "2016-05-18",  # Time Warner Cable acquired by Charter
    "TYC": "2016-09-06",  # Tyco merged into Johnson Controls
    "WFR": "2016-09-28",  # SunEdison Semiconductor / MEMC
    # --- 2015 ---
    "SIAL": "2015-11-18",  # Sigma-Aldrich acquired by Merck KGaA
    "ACT": "2015-06-15",  # Actavis renamed to Allergan plc (became AGN)
    "DTV": "2015-07-24",  # DirecTV acquired by AT&T
    "KRFT": "2015-07-02",  # Kraft Foods Group merged with Heinz -> KHC
    # --- 2014 ---
    "FRX": "2014-07-01",  # Forest Labs acquired by Actavis
    "LIFE": "2014-02-03",  # Life Technologies acquired by Thermo Fisher
    # --- 2013 ---
    "HNZ": "2013-06-07",  # H.J. Heinz acquired by Berkshire/3G
    # --- 2011 ---
    "MFE": "2011-02-25",  # McAfee acquired by Intel
    # --- 2010 and earlier ---
    "FNM": "2010-06-16",  # Fannie Mae delisted (conservatorship)
    "FRE": "2010-06-16",  # Freddie Mac delisted (same)
    "BUD": "2008-11-18",  # Anheuser-Busch acquired by InBev
    "CCR": "2008-07-01",  # Countrywide Financial acquired by Bank of America
    "LEH": "2008-09-15",  # Lehman Brothers bankruptcy
    "NCC": "2008-12-31",  # National City Corp acquired by PNC
}
