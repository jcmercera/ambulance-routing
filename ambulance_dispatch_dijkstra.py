###import libraries


import time
import pandas as pd
import heapq


### Load data from CSV files using pandas
ambulances = pd.read_csv("ambulance.csv")
calls = pd.read_csv("calls.csv")
roads = pd.read_csv("location_network.csv")
call_priority = pd.read_csv("call_priority.csv")

### Rename columns of CSV files for clarity and variable names in the code
ambulances.rename(columns={
    'Ambulance Number': 'AmbulanceID',
    'Staging Location': 'StagingLocation'
}, inplace=True)

### Set ambulances as available and start at their staging location
ambulances['CurrentLocation'] = ambulances['StagingLocation']       ###sets ambulance's current location to its home base
ambulances['Available'] = True          ###mark all ambulances available


### create call type priority
priority_map = dict(zip(call_priority["Call Type"], call_priority["Priority"]))

### apply priority and sort the calls
calls["Priority"] = calls["Call Type"].map(priority_map)
calls["OriginalOrder"] = calls.index
calls.sort_values(by=["Priority", "OriginalOrder"], inplace=True)
calls.reset_index(drop=True, inplace=True)

### road network graph with traffic delays included
graph = {}
edge_set = set()
for index, row in roads.iterrows():
    start, end = row['Start'], row['End']
    distance = row['Distance']
    delay = row['Traffic Delay']
    speed = row.get('Speed Limit', 40)  # default if missing
    travel_time = (distance / speed) * 60 + delay
    edge_key = tuple(sorted((start, end)))

    if edge_key in edge_set:
        continue
    edge_set.add(edge_key)

    graph.setdefault(start, []).append((end, travel_time))
    graph.setdefault(end, []).append((start, travel_time))


###FUNCTION to find the shortest route using Dijkstra's algorithm (double check geeksforgeeks)
def dijkstra(graph, start, goal):
    start_timer = time.perf_counter() ###start timer for dispatch
    queue = [(0, start, [])]    ###queue of nodes to visit
    visited = set()             ### set of visited locations

    while queue:
        total_cost, node, path = heapq.heappop(queue)            ### Gets the next shortest-distance node to explore
        if node in visited:
            continue
        visited.add(node)
        path = path + [node]                     ### Skips if already visited; otherwise, adds to the path.

        if node == goal:
            end_timer = time.perf_counter()   ###end dispatch timer
            return total_cost, path, (end_timer - start_timer) *1000  ###calculate dispatch time in ms               ### If goal is reached, return the total time and the path

        for neighbor, weight in graph.get(node, []):
            if neighbor not in visited:
                heapq.heappush(queue, (total_cost + weight, neighbor, path))     ### Checks all neighbor nodes and adds them to the queue with cumulative cost
    end_timer = time.perf_counter()
    return float('inf'), [], (end_timer - start_timer) * 1000        ### If no path found, return infinite distance and empty path

###prepare log file
log_file = "ambulance_call_log.csv"
with open(log_file, 'w') as log:
    log.write("")

### dispatch simulation
dispatch_log = []
route_times = []


### Process each emergency call
for index, call in calls.iterrows():        ###Loop over each call
    call_id = call['Call ID']
    call_type = call['Call Type']
    call_location = call['Location']             ###get the call details one at a time (see if this can be simplified)
    available_amb = ambulances[ambulances["Available"] == True]     ###find the best ambulance


    best_cost = float('inf')		###set up variables for best ambu
    best_amb = None
    best_path = []
    best_staging = None
    route_time = 0

    for i, amb in available_amb.iterrows():  ###loop through all available ambus
        amb_id = amb['AmbulanceID']
        current = amb['CurrentLocation']
        staging = amb['StagingLocation']
        cost, path, rt = dijkstra(graph, current, call_location)

        if cost < best_cost:		###find fastest ambu to respond
            best_cost = cost
            best_amb = amb_id
            best_path = path
            best_staging = staging
            route_time = rt

    if best_amb:
        ### Reset ambu to staging location after dispatch
        ambulances.loc[ambulances['AmbulanceID'] == best_amb, 'Available'] = False
        ambulances.loc[ambulances['AmbulanceID'] == best_amb, 'Available'] = True
        ambulances.loc[ambulances['AmbulanceID'] == best_amb, 'CurrentLocation'] = best_staging

        ### Create dispatch log entry, store key information
        dispatch = {
            'CallID': call_id,
            'CallType': call_type,
            'CallLocation': call_location,
            'SelectedAmbulance': best_amb,
            'Route': "->".join(best_path),
            'TimeToLocation': f"{best_cost:4f}",
            'RouteExecutionTime(ms)': f"{route_time:4f}"
        }
        ##append csv log
        dispatch_log.append(dispatch)
        log_entry = ",".join([f"{k}={v}" for k, v in dispatch.items()])
        with open(log_file, 'a') as log:
            log.write(log_entry + "\n")

        route_times.append(route_time)

### Summary log row
avg_time = sum(route_times) / len(route_times)if route_times else 0
dispatch_log.append({
    'CallID': 'SUMMARY',
    'CallType': '-',
    'CallLocation': '-',
    'SelectedAmbulance': '-',
    'Route': "->".join('-'),
    'TimeToLocation': '-',
    'RouteExecutionTime(ms)': f"{avg_time:4f}"
})


# Performance summary/print
# Performance summary/print
print("==== Performance Summary ====")

total_time = sum(route_times)
print(f"Route-finding execution time: {total_time:.4f} ms")

if route_times:
    avg_time = total_time / len(route_times)
    print(f"Average route-finding time: {avg_time:.4f} ms")
else:
    print("No calls were dispatched.")

