
from typing import Optional
from bin import pretty_print, solve_k_best
from bin_loop import pick_best_by_makespan_then_avgjct
from collections import defaultdict, deque


available_MIG_profile_per_binpacking_profile = {
    # No reserved instance
    "43": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 4g×1 + 3g×1
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 4g×1 + 2g×1 + 1g×1
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 4g×1 + 1g×3
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 2g×2
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "322": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 4g×1 + 3g×1
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 4g×1 + 2g×1 + 1g×1
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 4g×1 + 1g×3
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 2g×2
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "1111111": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 4g×1 + 3g×1
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 4g×1 + 2g×1 + 1g×1
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 4g×1 + 1g×3
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 2g×2
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    }, 
    "3211":{
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 3g×2
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 3g×1 + 1g×4
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 3g×1 + 1g×4
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 1g×4
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "421": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 4g×1 + 3g×1
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 4g×1 + 2g×1 + 1g×1
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 4g×1 + 1g×3
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 2g×2
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "4111": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 4g×1 + 3g×1
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 4g×1 + 2g×1 + 1g×1
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 4g×1 + 1g×3
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 2g×2
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "211111": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 4g×1 + 3g×1
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 4g×1 + 2g×1 + 1g×1
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 4g×1 + 1g×3
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 2g×2
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "3211": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0},
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0},
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},
    },
    "31111": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},  # 7g×1
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},  # 4g×1 + 3g×1
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0  },  # 4g×1 + 2g×1 + 1g×1
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0  },  # 4g×1 + 1g×3
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},  # 3g×1 + 1g×4
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},  # 3g×1 + 2g×2
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "1111111": {
        # profile with max 7g
        "7": {"3g": 0, "1g": 0,  "2g":0, "4g":0,"7g":1},
        # profile with max 4g
        "43": {"3g": 1, "1g": 0, "2g":0, "4g":1,"7g":0},
        "421": {"3g": 0, "1g": 1, "2g":1,"4g":1,"7g":0},
        "4111": {"3g": 0, "1g": 3, "2g":0, "4g":1,"7g":0},
        # profile with max 3g
        "31111": {"3g": 1, "1g": 4, "2g":0, "4g":0,"7g":0},
        "322":   {"3g": 1, "1g": 0, "2g":2, "4g":0,"7g":0},
        "3211":   {"3g": 1, "1g": 2, "2g":1, "4g":0,"7g":0},
        # profile with max 2g
        "22111":   {"3g": 0, "1g": 3, "2g":2, "4g":0,"7g":0},
        "2221":   {"3g": 0, "1g": 1, "2g":3, "4g":0,"7g":0},
        "211111":   {"3g": 0, "1g": 5, "2g":1, "4g":0,"7g":0},
        # profile  with max 1g
        "1111111": {"3g": 0, "1g": 7, "2g":0, "4g":0,"7g":0},
    },
    # # With 1G reserved 
    "111111": {
        # profile with max 7g
        # profile with max 4g
        # "42": {"3g": 1, "1g": 0, "2g":1, "4g":1,"7g":0},  # o
        # "411": {"3g": 0, "1g": 2, "2g":0, "4g":1,"7g":0  },  # o
        # profile with max 3g
        "3111": {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0}, # x # 3g×1 + 1g×4
        "321":   {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0}, # X # 3g×1 + 2g×2
        "3111":   {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "222":   {"3g": 0, "1g": 0, "2g":3, "4g":0,"7g":0}, # o
        "2211":   {"3g": 0, "1g": 2, "2g":2, "4g":0,"7g":0}, # 0
        "21111":   {"3g": 0, "1g": 4, "2g":1, "4g":0,"7g":0}, #0 
        "111111": {"3g": 0, "1g": 6, "2g":0, "4g":0,"7g":0},  # 1g×7
    }, 
    "42": {
        # profile with max 7g
        # profile with max 4g
        "42": {"3g": 1, "1g": 0, "2g":1, "4g":1,"7g":0},  # o
        "411": {"3g": 0, "1g": 2, "2g":0, "4g":1,"7g":0  },  # o
        # profile with max 3g
        "3111": {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0}, # x # 3g×1 + 1g×4
        "321":   {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0}, # X # 3g×1 + 2g×2
        "3111":   {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "222":   {"3g": 0, "1g": 0, "2g":3, "4g":0,"7g":0}, # o
        "2211":   {"3g": 0, "1g": 2, "2g":2, "4g":0,"7g":0}, # 0
        "21111":   {"3g": 0, "1g": 4, "2g":1, "4g":0,"7g":0}, #0 
        "111111": {"3g": 0, "1g": 6, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "411": {
        # profile with max 7g
        # profile with max 4g
        "42": {"3g": 1, "1g": 0, "2g":1, "4g":1,"7g":0},  # o
        "411": {"3g": 0, "1g": 2, "2g":0, "4g":1,"7g":0  },  # o
        # profile with max 3g
        "3111": {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0}, # x # 3g×1 + 1g×4
        "321":   {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0}, # X # 3g×1 + 2g×2
        "3111":   {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "222":   {"3g": 0, "1g": 0, "2g":3, "4g":0,"7g":0}, # o
        "2211":   {"3g": 0, "1g": 2, "2g":2, "4g":0,"7g":0}, # 0
        "21111":   {"3g": 0, "1g": 4, "2g":1, "4g":0,"7g":0}, #0 
        "111111": {"3g": 0, "1g": 6, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "3111": {
        # profile with max 7g
        # profile with max 4g
        "42": {"3g": 1, "1g": 0, "2g":1, "4g":1,"7g":0},  # o
        "411": {"3g": 0, "1g": 2, "2g":0, "4g":1,"7g":0  },  # o
        # profile with max 3g
        "3111": {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0}, # x # 3g×1 + 1g×4
        "321":   {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0}, # X # 3g×1 + 2g×2
        "3111":   {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0},  # 3g×1 + 2g×1 + 1g×2
        # profile with max 2g
        "222":   {"3g": 0, "1g": 0, "2g":3, "4g":0,"7g":0}, # o
        "2211":   {"3g": 0, "1g": 2, "2g":2, "4g":0,"7g":0}, # 0
        "21111":   {"3g": 0, "1g": 4, "2g":1, "4g":0,"7g":0}, #0 
        "111111": {"3g": 0, "1g": 6, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "321": {
        # profile with max 7g
        # profile with max 4g
        "42": {"3g": 1, "1g": 0, "2g":1, "4g":1,"7g":0},  # o
        "411": {"3g": 0, "1g": 2, "2g":0, "4g":1,"7g":0  },  # o
        # profile with max 3g
        "3111": {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0}, # x # 3g×1 + 1g×4
        "321":   {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0}, # X # 3g×1 + 2g×2
        # profile with max 2g
        "222":   {"3g": 0, "1g": 0, "2g":3, "4g":0,"7g":0}, # o
        "2211":   {"3g": 0, "1g": 2, "2g":2, "4g":0,"7g":0}, # 0
        "21111":   {"3g": 0, "1g": 4, "2g":1, "4g":0,"7g":0}, #0 
        "111111": {"3g": 0, "1g": 6, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "222": {        
        # profile with max 7g
        # profile with max 4g
        "42": {"3g": 1, "1g": 0, "2g":1, "4g":1,"7g":0},  # o
        "411": {"3g": 0, "1g": 2, "2g":0, "4g":1,"7g":0  },  # o
        # profile with max 3g
        "3111": {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0}, # x # 3g×1 + 1g×4
        "321":   {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0}, # X # 3g×1 + 2g×2
        # profile with max 2g
        "222":   {"3g": 0, "1g": 0, "2g":3, "4g":0,"7g":0}, # o
        "2211":   {"3g": 0, "1g": 2, "2g":2, "4g":0,"7g":0}, # 0
        "21111":   {"3g": 0, "1g": 4, "2g":1, "4g":0,"7g":0}, #0 
        "111111": {"3g": 0, "1g": 6, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "2211": {
        # profile with max 7g
        # profile with max 4g
        "42": {"3g": 1, "1g": 0, "2g":1, "4g":1,"7g":0},  # o
        "411": {"3g": 0, "1g": 2, "2g":0, "4g":1,"7g":0  },  # o
        # profile with max 3g
        "3111": {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0}, # x # 3g×1 + 1g×4
        "321":   {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0}, # X # 3g×1 + 2g×2
        # profile with max 2g
        "222":   {"3g": 0, "1g": 0, "2g":3, "4g":0,"7g":0}, # o
        "2211":   {"3g": 0, "1g": 2, "2g":2, "4g":0,"7g":0}, # 0
        "21111":   {"3g": 0, "1g": 4, "2g":1, "4g":0,"7g":0}, #0 
        "111111": {"3g": 0, "1g": 6, "2g":0, "4g":0,"7g":0},  # 1g×7
    },
    "21111": {
        # profile with max 7g
        # profile with max 4g
        "42": {"3g": 1, "1g": 0, "2g":1, "4g":1,"7g":0},  # o
        "411": {"3g": 0, "1g": 2, "2g":0, "4g":1,"7g":0  },  # o
        # profile with max 3g
        "3111": {"3g": 1, "1g": 3, "2g":0, "4g":0,"7g":0}, # x # 3g×1 + 1g×4
        "321":   {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0}, # X # 3g×1 + 2g×2
        # profile with max 2g
        "222":   {"3g": 0, "1g": 0, "2g":3, "4g":0,"7g":0}, # o
        "2211":   {"3g": 0, "1g": 2, "2g":2, "4g":0,"7g":0}, # 0
        "21111":   {"3g": 0, "1g": 4, "2g":1, "4g":0,"7g":0}, #0 
        "111111": {"3g": 0, "1g": 6, "2g":0, "4g":0,"7g":0},  # 1g×7
    },

    # With 2G reserved 
    "11111": {
        # profile with max 7g
        # profile with max 4g
        "32": {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0},  # o
        "311": {"3g": 1, "1g": 2, "2g":0, "4g":0,"7g":0  },  # o
        "221": {"3g": 0, "1g": 1, "2g":2, "4g":0,"7g":0  },  # o
        
    }, 
    # With 4G reserved 
    "111": {
        "111": {"3g": 0, "1g": 3, "2g":0, "4g":0,"7g":0  },  # 1g×7
        "3":        {"3g": 1, "1g": 0,  "2g":0, "4g":0,"7g":0 },  # 3g×2
        "21":   {"3g": 0, "1g": 1, "2g":1,"4g":0,"7g":0  },  # 3g×1 + 1g×4
    },
    "3": {
        # "4":    {"3g": 0, "1g": 0, "2g":0, "4g":1, "7g":0},  # 4g×1
         "3": {"3g": 1, "1g": 0, "2g":0, "4g":0,"7g":0 }, 
        # "1111": {"3g": 0, "1g": 4, "2g":0, "4g":0, "7g":0},  # 1g×4
        # "22":   {"3g": 0, "1g": 0, "2g":2, "4g":0, "7g":0},  # 2g×2
        # "211":  {"3g": 0, "1g": 2, "2g":1, "4g":0, "7g":0},  # 2g×1 + 1g×2
        "21":   {"3g": 0, "1g": 1, "2g":1, "4g":0, "7g":0},  # 2g×1 + 1g×1
        "111":  {"3g": 0, "1g": 3, "2g":0, "4g":0, "7g":0},  # 1g×3
        # "11":   {"3g": 0, "1g": 2, "2g":0, "4g":0, "7g":0},  # 1g×2
    },
    "21": {
        # "4":    {"3g": 0, "1g": 0, "2g":0, "4g":1, "7g":0},  # 4g×1
         "3": {"3g": 1, "1g": 0, "2g":0, "4g":0,"7g":0 }, 
        # "1111": {"3g": 0, "1g": 4, "2g":0, "4g":0, "7g":0},  # 1g×4
        # "22":   {"3g": 0, "1g": 0, "2g":2, "4g":0, "7g":0},  # 2g×2
        # "211":  {"3g": 0, "1g": 2, "2g":1, "4g":0, "7g":0},  # 2g×1 + 1g×2
        "21":   {"3g": 0, "1g": 1, "2g":1, "4g":0, "7g":0},  # 2g×1 + 1g×1
        "111":  {"3g": 0, "1g": 3, "2g":0, "4g":0, "7g":0},  # 1g×3
        # "11":   {"3g": 0, "1g": 2, "2g":0, "4g":0, "7g":0},  # 1g×2
    },
    # With 6G reserved
    "1": {
        "1": {"3g": 0, "1g": 1, "2g":0, "4g":0,"7g":0  },  # 1g×1
    },
    # With 3G reserved (4g available)
    "1111": {
        # "3": {"3g": 1, "1g": 0, "2g":0, "4g":0,"7g":0 },
        "1111": {"3g": 0, "1g": 4, "2g":0, "4g":0,"7g":0 },  # 1g×4
        "4": {"3g": 0, "1g": 0,  "2g":0 , "4g":1,"7g":0},  # 4g×1
        "22": {"3g": 0, "1g": 0,  "2g":2 , "4g":0,"7g":0},  # 2g×2
        "211": {"3g": 0, "1g": 2, "2g":1, "4g":0, "7g":0},  # 2g×1 + 1g×2
        "31": {"3g": 1, "1g": 1, "2g":0,"4g":0,"7g":0  },  # X
    },
    "4": {
        "1111": {"3g": 0, "1g": 4, "2g":0, "4g":0,"7g":0 },  # 1g×4
        "4": {"3g": 0, "1g": 0,  "2g":0 , "4g":1,"7g":0},  # 4g×1
        "22": {"3g": 0, "1g": 0,  "2g":2 , "4g":0,"7g":0},  # 2g×2
        "211": {"3g": 0, "1g": 2, "2g":1, "4g":0, "7g":0},  # 2g×1 + 1g×2
        "31": {"3g": 1, "1g": 1, "2g":0,"4g":0,"7g":0  },  # 2g×1 + 1g×1
    },
    "22": {
        "1111": {"3g": 0, "1g": 4, "2g":0, "4g":0,"7g":0 },  # 1g×4
        "4": {"3g": 0, "1g": 0,  "2g":0 , "4g":1,"7g":0},  # 체크 필요
        "22": {"3g": 0, "1g": 0,  "2g":2 , "4g":0,"7g":0},  # 2g×2
        "211": {"3g": 0, "1g": 2, "2g":1, "4g":0, "7g":0},  # 2g×1 + 1g×2
        "31": {"3g": 1, "1g": 1, "2g":0,"4g":0,"7g":0  },  # X
    },
    "211": {
        "1111": {"3g": 0, "1g": 4, "2g":0, "4g":0,"7g":0 },  # 1g×4
        "4": {"3g": 0, "1g": 0,  "2g":0 , "4g":1,"7g":0},  # X
        "22": {"3g": 0, "1g": 0,  "2g":2 , "4g":0,"7g":0},  # 2g×2
        "211": {"3g": 0, "1g": 2, "2g":1, "4g":0, "7g":0},  # 2g×1 + 1g×2
        "31": {"3g": 1, "1g": 1, "2g":0,"4g":0,"7g":0  },  # X
    },
    "31": {
        "1111": {"3g": 0, "1g": 4, "2g":0, "4g":0,"7g":0 },  # 0
        "4": {"3g": 0, "1g": 0,  "2g":0 , "4g":1,"7g":0},  # X
        "22": {"3g": 0, "1g": 0,  "2g":2 , "4g":0,"7g":0},  # X
        "211": {"3g": 0, "1g": 2, "2g":1, "4g":0, "7g":0},  # 0
        "31": {"3g": 1, "1g": 1, "2g":0,"4g":0,"7g":0  },  # 2g×1 + 1g×1
    },
    # With 2G reserved (5g available)
    "32": {
        "32": {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0},
        "311": {"3g": 1, "1g": 2, "2g":0, "4g":0,"7g":0},
        "221": {"3g": 0, "1g": 1, "2g":2, "4g":0,"7g":0},
        "2111": {"3g": 0, "1g": 3, "2g":1, "4g":0,"7g":0},
        "11111": {"3g": 0, "1g": 5, "2g":0, "4g":0,"7g":0},
    },
    "311": {
        "32": {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0},
        "311": {"3g": 1, "1g": 2, "2g":0, "4g":0,"7g":0},
        "221": {"3g": 0, "1g": 1, "2g":2, "4g":0,"7g":0},
        "2111": {"3g": 0, "1g": 3, "2g":1, "4g":0,"7g":0},
        "11111": {"3g": 0, "1g": 5, "2g":0, "4g":0,"7g":0},

    },
    "221": {    
                "32": {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0},
        "311": {"3g": 1, "1g": 2, "2g":0, "4g":0,"7g":0},
        "221": {"3g": 0, "1g": 1, "2g":2, "4g":0,"7g":0},
        "2111": {"3g": 0, "1g": 3, "2g":1, "4g":0,"7g":0},
        "11111": {"3g": 0, "1g": 5, "2g":0, "4g":0,"7g":0},
    },
    "2111": {
        "32": {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0},
        "311": {"3g": 1, "1g": 2, "2g":0, "4g":0,"7g":0},
        "221": {"3g": 0, "1g": 1, "2g":2, "4g":0,"7g":0},
        "2111": {"3g": 0, "1g": 3, "2g":1, "4g":0,"7g":0},
        "11111": {"3g": 0, "1g": 5, "2g":0, "4g":0,"7g":0},
    },
    "11111": {
        "32": {"3g": 1, "1g": 0, "2g":1, "4g":0,"7g":0},
        "311": {"3g": 1, "1g": 2, "2g":0, "4g":0,"7g":0},
        "221": {"3g": 0, "1g": 1, "2g":2, "4g":0,"7g":0},
        "2111": {"3g": 0, "1g": 3, "2g":1, "4g":0,"7g":0},
        "11111": {"3g": 0, "1g": 5, "2g":0, "4g":0,"7g":0},
    },

    # With 5G reserved (2g available)   
    "11": {
        "2": {"3g": 0, "1g": 0, "2g":1, "4g":0, "7g":0},  # 2g×1
        "11": {"3g": 0, "1g": 2, "2g":0, "4g":0, "7g":0},  # 1g×2
    },
    "2": {
        "2": {"3g": 0, "1g": 0, "2g":1, "4g":0, "7g":0},  # 2g×1
        "11": {"3g": 0, "1g": 2, "2g":0, "4g":0, "7g":0},  # 1g×2
    },
}





def run_bin_pack(
        init_config: str = "1111",
        job_list: Optional[list] = None,
        slot_minutes: int = 10,
        delta: float = 0.1,
        # === 하이브리드 로직용 새 파라미터 ===
        reserved_profile: Optional[str] = None,  # HP가 사용 중인 프로파일 (예: "3", "31")
        node_name: Optional[str] = None,         # 노드 이름 (nvidia-smi 원격 실행용)
        gpu_index: int = 0,                      # GPU 인덱스
        # === Cleanup 모드 (실제 삭제 후 조회) ===
        cleanup_mode: bool = False,              # True: managed 인스턴스 삭제 후 정확한 조회
        reserved_gi_ids: Optional[list] = None,  # 삭제하지 않을 GI ID 리스트 (HP용)
        # === Dry-run 모드 ===
        dry_run: bool = False,                   # True: 동적 조회 없이 정적 configs만 사용
    ):
    """
    MIG 구성 최적화를 위한 ILP 기반 Bin Packing

    Args:
        init_config: 현재 사용 가능한 MIG 프로파일 (HP 제외)
        job_list: 스케줄링할 Job 리스트 [{"name": ..., "size": ..., "duration": ...}, ...]
        slot_minutes: 슬롯 길이 (분)
        delta: MIG 재구성 페널티
        reserved_profile: HP job이 사용 중인 프로파일 (예: "3" → 3g 사용 중)
                         None이면 정적 딕셔너리 사용, 있으면 nvidia-smi 동적 조회
        node_name: 노드 이름 (원격 실행용, None이면 로컬)
        gpu_index: GPU 인덱스
        cleanup_mode: True면 managed 인스턴스를 실제로 삭제한 후 조회 (가장 정확)
        reserved_gi_ids: cleanup_mode=True일 때, 삭제하지 않을 GI ID 리스트

    Returns:
        ILP 결과 딕셔너리 또는 None

    모드:
        - reserved_profile == None: 정적 딕셔너리 사용 (빠름)
        - cleanup_mode == True: 실제 삭제 후 조회 (가장 정확, placement 반영)
        - cleanup_mode == False + reserved_profile != None: total 기반 조회 (fallback)
    """
    configs = None

    # ================================================================
    # 사전 검사: reserved_profile이 전체 GPU를 사용하면 early return
    # ================================================================
    if reserved_profile:
        reserved_capacity = sum(int(c) for c in reserved_profile)
        if reserved_capacity >= 7:
            print(f"    [bin_pack] HP jobs using full GPU ({reserved_capacity}g >= 7g)")
            print(f"    [bin_pack] No space for Spot jobs, returning None")
            return None

    # ================================================================
    # Dry-run 모드: 동적 조회 없이 바로 정적 configs 사용
    # ================================================================
    if dry_run:
        reserved_capacity = sum(int(c) for c in reserved_profile) if reserved_profile else 0
        available_capacity = 7 - reserved_capacity
        static_key = "1" * available_capacity

        print(f"    [bin_pack] DRY-RUN mode: Using STATIC configs")
        print(f"    [bin_pack] Reserved: {reserved_profile or 'None'} ({reserved_capacity}g), Available: {available_capacity}g")

        if static_key in available_MIG_profile_per_binpacking_profile:
            configs = available_MIG_profile_per_binpacking_profile[static_key]
            print(f"    [bin_pack] Static key: '{static_key}' → {len(configs)} configs")
        else:
            print(f"    [bin_pack] WARNING: Static key '{static_key}' not found, trying init_config '{init_config}'")
            if init_config in available_MIG_profile_per_binpacking_profile:
                configs = available_MIG_profile_per_binpacking_profile[init_config]

    # ================================================================
    # Cleanup 모드: 실제 삭제 후 정확한 조회 (가장 정확)
    # ================================================================
    if not dry_run and cleanup_mode and node_name:
        print(f"    [bin_pack] Using CLEANUP mode (most accurate)")
        try:
            from mig_dynamic_config import get_dynamic_configs_with_cleanup, validate_configs

            configs = get_dynamic_configs_with_cleanup(
                node_name=node_name,
                gpu_index=gpu_index,
                reserved_gi_ids=reserved_gi_ids or [],
            )

            # "In use by another client" 에러 감지
            if isinstance(configs, dict) and configs.get("IN_USE_ERROR"):
                print(f"    [bin_pack] 'In use by another client' error detected!")
                return {"IN_USE_ERROR": True}

            if configs and validate_configs(configs):
                print(f"    [bin_pack] Cleanup mode: {len(configs)} configs (placement-aware)")
            else:
                print(f"    [bin_pack] WARNING: Cleanup mode failed, falling back")
                configs = None

        except Exception as e:
            print(f"    [bin_pack] ERROR: Cleanup mode failed: {e}")
            configs = None

    # ================================================================
    # 하이브리드 로직: Reserved 여부에 따라 정적/동적 선택 (dry_run이 아닐 때만)
    # ================================================================
    if not dry_run and configs is None and reserved_profile is None:
        # Case 1: Reserved 없음 → 정적 딕셔너리 사용
        if init_config in available_MIG_profile_per_binpacking_profile:
            configs = available_MIG_profile_per_binpacking_profile[init_config]
            print(f"    [bin_pack] Using STATIC configs for '{init_config}' (no reserved)")
        else:
            print(f"    [bin_pack] WARNING: init_config '{init_config}' not in static dict, trying dynamic")
            reserved_profile = ""  # fallback to dynamic

    if not dry_run and configs is None and reserved_profile is not None:
        # Case 2: Reserved 있음 → nvidia-smi 동적 조회 (total 기반)
        try:
            from mig_dynamic_config import get_dynamic_configs, validate_configs

            # reserved_profile에서 슬라이스 수 계산 (예: "31" → 4)
            reserved_slices = sum(int(c) for c in reserved_profile) if reserved_profile else 0

            print(f"    [bin_pack] Using DYNAMIC configs (reserved='{reserved_profile}', {reserved_slices}g)")
            print(f"    [bin_pack] Querying nvidia-smi on node={node_name}, gpu={gpu_index}")

            configs = get_dynamic_configs(
                node_name=node_name,
                gpu_index=gpu_index,
                reserved_slices=reserved_slices,
            )

            if configs:
                # 유효성 검사
                if validate_configs(configs):
                    print(f"    [bin_pack] Dynamic configs loaded: {len(configs)} profiles")
                    for cfg_name in sorted(configs.keys())[:5]:  # 처음 5개만 출력
                        print(f"      - {cfg_name}: {configs[cfg_name]}")
                    if len(configs) > 5:
                        print(f"      ... and {len(configs) - 5} more")
                else:
                    print(f"    [bin_pack] WARNING: Dynamic configs validation failed, falling back")
                    configs = None
            else:
                print(f"    [bin_pack] WARNING: No dynamic configs returned")
                configs = None  # fallback을 위해 None으로 설정

        except ImportError as e:
            print(f"    [bin_pack] ERROR: Failed to import mig_dynamic_config: {e}")
            configs = None
        except Exception as e:
            print(f"    [bin_pack] ERROR: Dynamic config failed: {e}")
            configs = None

        # Fallback: 동적 조회 실패 시 정적 딕셔너리 시도
        if configs is None or len(configs) == 0:
            print(f"    [bin_pack] Falling back to STATIC configs")
            if init_config in available_MIG_profile_per_binpacking_profile:
                configs = available_MIG_profile_per_binpacking_profile[init_config]
            else:
                # 마지막 fallback: HP reserved를 고려한 프로파일
                reserved_capacity = sum(int(c) for c in reserved_profile) if reserved_profile else 0
                available_capacity = 7 - reserved_capacity
                fallback_key = "1" * available_capacity  # e.g., "111111" for 6g available

                if fallback_key in available_MIG_profile_per_binpacking_profile:
                    configs = available_MIG_profile_per_binpacking_profile[fallback_key]
                    print(f"    [bin_pack] Using fallback '{fallback_key}' (reserved={reserved_profile}, {reserved_capacity}g)")

    # ================================================================
    # ILP 실행
    # ================================================================
    if configs is None or len(configs) == 0:
        print(f"    [bin_pack] ERROR: No valid configs available")
        return None

    sols = solve_k_best(5, job_list, configs, slot_minutes, delta, init_config)
    best = pick_best_by_makespan_then_avgjct(sols, slot_minutes)

    if best:
        print("\n=== Bin Packing Result ===")
        pretty_print(best, slot_minutes, init_config)

    return best

def mock_bin_pack(queued_jobs, current_profile):
    """간단한 휴리스틱 기반 MIG profile 선택"""
    # Job 크기별 개수 계산
    req_counts = defaultdict(int)
    for job in queued_jobs:
        req_counts[job["size"]] += 1

    # 휴리스틱: 가장 많은 크기에 맞춰 profile 선택
    if req_counts[3] >= 2:
        # 3g job이 2개 이상이면 "3211"
        return {"chosen_cfg": {1: "3211"}, "avg_jct": 15.0}
    elif req_counts[2] >= 3:
        # 2g job이 3개 이상이면 "2221"
        return {"chosen_cfg": {1: "2221"}, "avg_jct": 18.0}
    else:
        # 기본값 유지
        return {"chosen_cfg": {1: current_profile}, "avg_jct": 20.0}

print("Mock bin packing function defined")