# MIG Configuration for NVIDIA H100 NVL
# Each config string represents a MIG partition configuration
# Format: "31111" means 1x3g + 4x1g (total 7g for H100)

mig_config = {
    # ===== 7g Total Configurations =====
    # Full GPU configurations (7g total for H100)
    "7": {
        "7g": 1,
        "4g": 0,
        "3g": 0,
        "2g": 0,
        "1g": 0,
    },

    # 3g + smaller slices (NOT USED - 3g reserved for critical workload)
    "3211": {
        "7g": 0,
        "4g": 0,
        "3g": 1,
        "2g": 1,
        "1g": 2,
    },
    "31111": {
        "7g": 0,
        "4g": 0,
        "3g": 1,
        "2g": 0,
        "1g": 4,
    },
    "322": {
        "7g": 0,
        "4g": 0,
        "3g": 1,
        "2g": 2,
        "1g": 0,
    },

    # 4g + smaller slices (7g total)
    "4111": {
        "7g": 0,
        "4g": 1,
        "3g": 0,
        "2g": 0,
        "1g": 3,
    },
    "421": {
        "7g": 0,
        "4g": 1,
        "3g": 0,
        "2g": 1,
        "1g": 1,
    },
    "43": {
        "7g": 0,
        "4g": 1,
        "3g": 1,
        "2g": 0,
        "1g": 0,
    },

    # 2g configurations (7g total)
    "2211": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 2,
        "1g": 3,
    },
    "222": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 3,
        "1g": 1,
    },
    "211111": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 1,
        "1g": 5,
    },

    # 1g only configurations (7g total)
    "1111111": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 0,
        "1g": 7,
    },
    "111111": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 0,
        "1g": 6,
    },
    "11111": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 0,
        "1g": 5,
    },

    # ===== 4g Total Configurations =====
    # 4g slice (4g total)
    "4": {
        "7g": 0,
        "4g": 1,
        "3g": 0,
        "2g": 0,
        "1g": 0,
    },

    # 2g configurations (4g total)
    "22": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 2,
        "1g": 0,
    },
    "211": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 1,
        "1g": 2,
    },
    "21": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 1,
        "1g": 1,
    },

    # 1g only configurations (4g total)
    "1111": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 0,
        "1g": 4,
    },
    "111": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 0,
        "1g": 3,
    },
    "11": {
        "7g": 0,
        "4g": 0,
        "3g": 0,
        "2g": 0,
        "1g": 2,
    },
}
