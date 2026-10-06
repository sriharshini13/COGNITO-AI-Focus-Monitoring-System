from modules.session_logger import SessionLogger

logger = SessionLogger()

# Simulate saving 3 sessions
sessions_data = [
    ({'High':300,'Medium':80,'Low':40},
     {'Engaged':300,'Bored':60,'Confused':40,'Frustrated':20},
     {'average_score':78,'peak_score':97,'grade':'B'},
     {'total_distractions':5,'distractions_per_hour':12.0},
     1500),

    ({'High':200,'Medium':100,'Low':100},
     {'Engaged':200,'Bored':100,'Confused':80,'Frustrated':20},
     {'average_score':62,'peak_score':88,'grade':'C'},
     {'total_distractions':8,'distractions_per_hour':19.2},
     1800),

    ({'High':400,'Medium':50,'Low':10},
     {'Engaged':400,'Bored':30,'Confused':20,'Frustrated':10},
     {'average_score':91,'peak_score':99,'grade':'A'},
     {'total_distractions':2,'distractions_per_hour':4.8},
     1200),
]

for data in sessions_data:
    logger.save_session(*data)

print("\n=== All Sessions ===")
for s in logger.get_all_sessions():
    print(f"  {s['date']} {s['time']} | "
          f"Productivity: {s['avg_productivity']} | "
          f"Grade: {s['grade']} | "
          f"Distractions: {s['total_distractions']}")

print("\n=== This Week Summary ===")
summary = logger.get_stats_summary(logger.get_this_week())
for k, v in summary.items():
    print(f"  {k}: {v}")