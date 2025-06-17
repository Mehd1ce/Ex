
1 .def acm_team(topic):
    max_topics = 0
    count = 0
    n = len(topic)
    for i in range(n):
        for j in range(i+1, n):
            combined = bin(int(topic[i], 2) | int(topic[j], 2)).count('1')
            if combined > max_topics:
                max_topics = combined
                count = 1
            elif combined == max_topics:
                count += 1
return [max_topics, count]

2.def alternate(s):
    unique = list(set(s))
    max_len = 0
    for i in range(len(unique)):
        for j in range(i+1, len(unique)):
            a, b = unique[i], unique[j]
            temp = [c for c in s if c == a or c == b]
            if all(temp[k] != temp[k+1] for k in range(len(temp)-1)):
                max_len = max(max_len, len(temp))
    return max_len

3.def are_sentences_similar(sentence1, sentence2, similar_pairs):
    if len(sentence1) != len(sentence2):
        return False
    pair_set = set((a, b) for a, b in similar_pairs)
return all(w1 == w2 or (w1, w2) in pair_set or (w2, w1) in pair_set for w1, w2 in zip(sentence1, sentence2))

4.def array_rank_transform(arr):
    sorted_arr = sorted(set(arr))
    rank_dict = {v: i+1 for i, v in enumerate(sorted_arr)}
return [rank_dict[v] for v in arr]

5.def average_salary_excluding_min_max(salary):
    return (sum(salary) - min(salary) - max(salary)) / (len(salary) - 2)

6.def balanced_sums(arr):
    total = sum(arr)
    left = 0
    for num in arr:
        if left == total - left - num:
            return "YES"
        left += num
    return "NO"

7.def beautifulDays(i, j, k):
return sum(abs(x - int(str(x)[::-1])) % k == 0 for x in range(i, j+1))

8.def buddy_strings(s, goal):
    if len(s) != len(goal):
        return False
    if s == goal and len(set(s)) < len(s):
        return True
    diffs = [(a, b) for a, b in zip(s, goal) if a != b]
return len(diffs) == 2 and diffs[0] == diffs[1][::-1]


9.def caesar_cipher(s, k):
    result = []
    for char in s:
        if char.isupper():
            result.append(chr((ord(char) - 65 + k) % 26 + 65))
        elif char.islower():
            result.append(chr((ord(char) - 97 + k) % 26 + 97))
        else:
            result.append(char)
return ''.join(result)


10.def calculate_typing_time(keyboard, word):
    pos = {c: i for i, c in enumerate(keyboard)}
return sum(abs(pos[word[i]] - pos[word[i-1]]) for i in range(1, len(word)))


11.def can_be_equal(target, arr):
return sorted(target) == sorted(arr)


12.def power_of_string(s):
    max_len = cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 1
    return max_len



13.def can_place_flowers(flowerbed, n):
    count = 0
    for i in range(len(flowerbed)):
        if flowerbed[i] == 0 and (i == 0 or flowerbed[i-1] == 0) and (i == len(flowerbed)-1 or flowerbed[i+1] == 0):
            flowerbed[i] = 1
            count += 1
return count >= n



14.def cavity_map(grid):
    grid = [list(row) for row in grid]
    for i in range(1, len(grid)-1):
        for j in range(1, len(grid[0])-1):
            if all(grid[i][j] > grid[x][y] for x, y in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]):
                grid[i][j] = 'X'
return [''.join(row) for row in grid]



15.def check_ones_distance(nums, k):
    last = -k - 1
    for i, num in enumerate(nums):
        if num == 1:
            if i - last - 1 < k:
                return False
            last = i
return True



16.def check_segment_length(s):
return all(len(seg) % 2 == 0 for seg in s.split('0') if seg)


17.def check_straight_line(coordinates):
    (x0, y0), (x1, y1) = coordinates[:2]
return all((y1 - y0) * (x - x0) == (y - y0) * (x1 - x0) for x, y in coordinates[2:])



18.def chocolate_feast(n, c, m):
    wrappers = chocolates = n // c
    while wrappers >= m:
        new = wrappers // m
        chocolates += new
        wrappers = wrappers % m + new
return chocolates

19.def closest_numbers(arr):
    arr.sort()
    min_diff = min(arr[i+1] - arr[i] for i in range(len(arr)-1))
    return [x for i in range(len(arr)-1) for x in (arr[i], arr[i+1]) if arr[i+1] - arr[i] == min_diff]



20.def coin_change_combinations(coins, amount):
    dp = [1] + [0] * amount
    for coin in coins:
        for i in range(coin, amount+1):
            dp[i] += dp[i-coin]
return dp[amount]



21.def compare_swaps_bubble_insertion_sort(arr):
    bubble, insertion = 0, 0
    # Bubble sort
    a = arr[:]
    for i in range(len(a)):
        for j in range(len(a) - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                bubble += 1
    # Insertion sort
    a = arr[:]
    for i in range(1, len(a)):
        j = i
        while j > 0 and a[j] < a[j - 1]:
            a[j], a[j - 1] = a[j - 1], a[j]
            insertion += 1
            j -= 1
return "insertion" if insertion <= bubble else "bubble"


22.def count_incremovable_subarrays(nums):
    count = 0
    for i in range(len(nums)):
        for j in range(i, len(nums)):
            temp = nums[:i] + nums[j+1:]
            if temp == sorted(temp):
                count += 1
return count



23.def count_negatives(grid):
    return sum(1 for row in grid for x in row if x < 0)


24.def count_special_problems(n, k, arr):
    special = 0
    page = 1
    for i in range(n):
        problems = arr[i]
        for p in range(1, problems+1):
            if p == page:
                special += 1
            if p % k == 0 or p == problems:
                page += 1
return special



25.def dist_plan_performance(calories, k, lower, upper):
    total = 0
    for i in range(len(calories)-k+1):
        s = sum(calories[i:i+k])
        if s < lower:
            total -= 1
        elif s > upper:
            total += 1
return total



26.def earliest_year_with_maximum_population(logs):
    years = [0] * 101  # 1950–2050
    for birth, death in logs:
        years[birth - 1950] += 1
        years[death - 1950] -= 1
    max_pop, year, pop = 0, 1950, 0
    for i in range(101):
        pop += years[i]
        if pop > max_pop:
            max_pop, year = pop, 1950 + i
    return year




27.def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
return dp[m][n]



28.def fair_candy_swap(alice_sizes, bob_sizes):
    sum_a, sum_b = sum(alice_sizes), sum(bob_sizes)
    target = (sum_a + sum_b) // 2
    set_b = set(bob_sizes)
    
    for x in alice_sizes:
        y = x + (target - sum_a)
        if y in set_b:
            return [x, y]
return []




30.def find_poisoned_duration(time_series, duration):
    if not time_series:
        return 0
    total = 0
    for i in range(1, len(time_series)):
        total += min(time_series[i] - time_series[i-1], duration)
return total + duration




31.def find_common_elements(arr1, arr2, arr3):
return sorted(list(set(arr1) & set(arr2) & set(arr3)))




32.def find_common_strings_with_least_index_sum(list1, list2):
    common = set(list1) & set(list2)
    if not common:
        return []
    min_sum = min(list1.index(x) + list2.index(x) for x in common)
return [x for x in common if list1.index(x) + list2.index(x) == min_sum]



33.def find_destination_city(paths):
    sources = set()
    destinations = set()
    for s, d in paths:
        sources.add(s)
        destinations.add(d)
return (destinations - sources).pop()



34.def find_find_array(arr):
return [arr[i] ^ arr[i-1] if i > 0 else arr[i] for i in range(len(arr))]



35.def find_judge(n, trust):
    count = [0] * (n + 1)
    for a, b in trust:
        count[a] -= 1
        count[b] += 1
    for i in range(1, n+1):
        if count[i] == n - 1:
            return i
return -1
36.def find_kth_positive(arr, k):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] - mid - 1 < k:
            left = mid + 1
        else:
            right = mid
    return left + k


37.def find_length_of_lcis(nums):
    max_len = curr = 1
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 1
return max_len



38.def find_max_sum_less_than_k(nums, k):
    max_sum = -1
    nums.sort() 
    
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum < k:
            max_sum = max(max_sum, current_sum)
            left += 1
        else:
            right -= 1
    
return max_sum



39.def find_nearest_valid_point(x, y, points):
    min_dist = float('inf')
    result = -1
    for i, (a, b) in enumerate(points):
        if a == x or b == y:
            dist = abs(x - a) + abs(y - b)
            if dist < min_dist:
                min_dist = dist
                result = i
return result



40.def find_missing_value(arr):
    n = len(arr)
    total_diff = (arr[-1] - arr[0]) // n  
    for i in range(1, n):
        if arr[i] - arr[i-1] != total_diff:
            return arr[i-1] + total_diff
return arr[-1] + total_diff


41.def find_shortest_subarray_with_degree(nums):
    left, right, count = {}, {}, {}
    for i, num in enumerate(nums):
        if num not in left:
            left[num] = i
        right[num] = i
        count[num] = count.get(num, 0) + 1
    degree = max(count.values())
return min(right[num] - left[num] + 1 for num in count if count[num] == degree)


42.def find_special_integer(nums):
    n = len(nums)
    for num in set(nums):
        if nums.count(num) > n / 4:
            return num
return -1


43.def find_the_distance_value(arr1, arr2, d):
return sum(all(abs(a - b) > d for b in arr2) for a in arr1)





44.def flood_fill(image, sr, sc, color):
    if image[sr][sc] == color:
        return image
    old_color = image[sr][sc]
    stack = [(sr, sc)]
    while stack:
        r, c = stack.pop()
        if 0 <= r < len(image) and 0 <= c < len(image[0]) and image[r][c] == old_color:            image[r][c] = color
            stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
return image


45.def format_with_dots(n):
    s = str(n)[::-1]
return '.'.join(s[i:i+3] for i in range(0, len(s), 3))[::-1]


46.def game_of_stones(n):
    return "First" if n % 7 > 1 else "Second"


47.from collections import Counter

def happy_lady_bugs(b):
    c = Counter(b)
    if c['_'] == 0:
        return all(b[i] == b[i+1] or b[i] == b[i-1] for i in range(1, len(b)-1))
return all(v > 1 or k == '_' for k, v in c.items())


48.def how_many_games(p, d, m, s):
    cnt = 0
    while s >= p:
        s -= p
        cnt += 1
        p = max(p - d, m)
return cnt





49.def icecream_parlor(m, cost):

    price_index = {}

    for i, price in enumerate(cost, 1):  
        complement = m - price
        if complement in price_index:
            return sorted([price_index[complement], i])
        price_index[price] = i

return []  


50.from collections import Counter

def intersect_two_arrays(a, b):
    c1, c2 = Counter(a), Counter(b)
    res = []
    for x in c1:
        res += [x] * min(c1[x], c2[x])
    return res


51.from collections import Counter

def isValidStringSameOccurence(s):
    v = list(Counter(s).values())
    return "YES" if v.count(min(v)) + v.count(max(v)) == len(v) and abs(max(v) - min(v)) <= 1 else "NO"


52.def is_majority_element(nums, target):
    return nums.count(target) > len(nums) // 2


53.def is_path_crossing(path):
    x = y = 0
    visited = {(0, 0)}
    moves = {'N': (0, 1), 'S': (0, -1), 'E': (1, 0), 'W': (-1, 0)}
    for d in path:
        dx, dy = moves[d]
        x += dx
        y += dy
        if (x, y) in visited:
            return True
        visited.add((x, y))
return False



54.def is_valid_mountain_array(arr):
    if len(arr) < 3:
        return False
    i = 0
    while i < len(arr)-1 and arr[i] < arr[i+1]:
        i += 1
    if i == 0 or i == len(arr)-1:
        return False
    while i < len(arr)-1 and arr[i] > arr[i+1]:
        i += 1
return i == len(arr)-1


55.def island_perimeter(grid):
    perimeter = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                perimeter += 4
                if i > 0 and grid[i-1][j] == 1:
                    perimeter -= 2
                if j > 0 and grid[i][j-1] == 1:
                    perimeter -= 2
return perimeter



56.def jumping_on_clouds(c):
    jumps = i = 0
    while i < len(c)-1:
        if i+2 < len(c) and c[i+2] == 0:
            i += 2
        else:
            i += 1
        jumps += 1
    return jumps


57.def k_closest(points, k):
return sorted(points, key=lambda x: x[0]**2 + x[1]**2)[:k]


58.def k_weakest_rows(mat, k):
return sorted(range(len(mat)), key=lambda x: (sum(mat[x]), x))[:k]


59.def kids_with_candies(candies, extra_candies):
    max_candy = max(candies)
return [c + extra_candies >= max_candy for c in candies]

60.def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(1, capacity+1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
return dp[n][capacity]

61.def largest_permutation(k, arr):
    pos = {v:i for i,v in enumerate(arr)}
    for i in range(len(arr)):
        if k == 0:
            break
        if arr[i] != len(arr)-i:
            j = pos[len(arr)-i]
            arr[i], arr[j] = arr[j], arr[i]
            pos[arr[i]], pos[arr[j]] = pos[arr[j]], pos[arr[i]]
            k -= 1
return arr



62.def largest_sum_after_k_negations(nums, k):
    nums.sort()
    for i in range(len(nums)):
        if k and nums[i] < 0:
            nums[i] = -nums[i]
            k -= 1
    if k % 2: nums.sort(); nums[0] = -nums[0]
    return sum(nums)


63.def longest_duration_key(keys, times):
    max_dur, res = times[0], keys[0]
    for i in range(1, len(times)):
        d = times[i] - times[i-1]
        if d > max_dur or (d == max_dur and keys[i] > res):
            max_dur, res = d, keys[i]
    return res





64.def longest_nice_substring(s):
    if not s:
        return ""
    unique = set(s)
    for i, c in enumerate(s):
        if c.swapcase() not in unique:
            left = longest_nice_substring(s[:i])
            right = longest_nice_substring(s[i+1:])
            return max(left, right, key=len)
return s




65.def longest_substring_between_equal_characters(s):
    first_pos = {}
    max_len = -1
    for i, c in enumerate(s):
        if c in first_pos:
            max_len = max(max_len, i - first_pos[c] - 1)
        else:
            first_pos[c] = i
return max_len


66.def luck_balance(k, contests):
    important = sorted([x[0] for x in contests if x[1] == 1], reverse=True)
    unimportant = [x[0] for x in contests if x[1] == 0]
    return sum(unimportant) + sum(important[:k]) - sum(important[k:])



67.def make_good(s):
    stack = []
    for c in s:
        if stack and abs(ord(stack[-1]) - ord(c)) == 32:
            stack.pop()
        else:
            stack.append(c)
return ''.join(stack)

68.def matrix_reshape(mat, r, c):
    if len(mat) * len(mat[0]) != r * c:
        return mat
    flat = [num for row in mat for num in row]
    return [flat[i*c:(i+1)*c] for i in range(r)]


69.from collections import Counter

def max_balloons(text):
    balloon_counts = Counter(text)
    required = {'b':1, 'a':1, 'l':2, 'o':2, 'n':1}
    
    max_num = float('inf')
    for char, count in required.items():
        max_num = min(max_num, balloon_counts.get(char, 0) // count)
    
return max_num




70.def max_subsequence_size(nums, queries):
    nums.sort()
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    
    result = []
    for q in queries:
        left, right = 0, len(nums)
        while left < right:
            mid = (left + right) // 2
            if prefix[mid] > q:
                right = mid
            else:
                left = mid + 1
        result.append(left - 1)
    
return result




71.def max_sum_less_than_k(nums, k):
    nums.sort()
    left, right = 0, len(nums)-1
    max_sum = -1
    while left < right:
        s = nums[left] + nums[right]
        if s < k:
            max_sum = max(max_sum, s)
            left += 1
        else:
            right -= 1
return max_sum

72.def maximum_wealth(accounts):
return max(sum(account) for account in accounts)


73.def migratory_birds(arr):
    from collections import Counter
    count = Counter(arr)
    max_count = max(count.values())
    most_common = [bird for bird in count if count[bird] == max_count]
    return min(most_common)


74.def min_operations_to_make_strictly_increasing(nums: list[int]) -> int:
    operations = 0
    for i in range(1, len(nums)):
        if nums[i] <= nums[i - 1]:
            operations += (nums[i - 1] + 1 - nums[i])
            nums[i] = nums[i - 1] + 1
    return operations



75.def min_path_cost(grid, move_cost):
    m, n = len(grid), len(grid[0])
    dp = [[0]*n for _ in range(m)]
    dp[0] = grid[0]
    for i in range(1, m):
        for j in range(n):
            dp[i][j] = min(dp[i-1][k] + move_cost[grid[i-1][k]][j] for k in range(n)) + grid[i][j]
return min(dp[-1])




76.def min_path_sum(grid):
    m, n = len(grid), len(grid[0])
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    for j in range(1, n):
        grid[0][j] += grid[0][j-1]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
return grid[-1][-1]





77.def minimum_distances(a):
    last_pos = {}
    min_dist = float('inf')
    for i, num in enumerate(a):
        if num in last_pos:
            min_dist = min(min_dist, i - last_pos[num])
        last_pos[num] = i
return min_dist if min_dist != float('inf') else -1



78.from collections import Counter

def missing_numbers(arr: list[int], brr: list[int]) -> list[int]:
    count_arr = Counter(arr)
    count_brr = Counter(brr)
    missing = []

    for num in count_brr:
        if count_brr[num] > count_arr.get(num, 0):
            missing.append(num)

    return sorted(missing)



79.def modify_string(s):
    s = list(s)
    for i in range(len(s)):
        if s[i] == '?':
            for c in 'abc':
                if (i == 0 or s[i-1] != c) and (i == len(s)-1 or s[i+1] != c):
                    s[i] = c
                    break
return ''.join(s)

80.def next_greatest_letter(letters, target):
    for c in letters:
        if c > target:
            return c
return letters[0]







81.def odd_cells(m, n, indices):
    rows = [0] * m
    cols = [0] * n
    for r, c in indices:
        rows[r] ^= 1
        cols[c] ^= 1
return sum((rows[i] + cols[j]) % 2 for i in range(m) for j in range(n))


82.from collections import Counter

def picking_numbers(a: list[int]) -> int:
    count = Counter(a)
    max_len = 0
    for num in count:
        max_len = max(max_len, count[num] + count.get(num + 1, 0))
    return max_len


83.def power_of_string(s):
    max_power = current = 1
    for i in range(1, len(s)):
        if s[i] == s[i-1]:
            current += 1
            max_power = max(max_power, current)
        else:
            current = 1
return max_power



84.def quick_sort_partition(arr, low, high):
    pivot = arr[high]
    i = low
    for j in range(low, high):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
return i





85.def reformat_string(s):
    letters = [c for c in s if c.isalpha()]
    digits = [c for c in s if c.isdigit()]
    if abs(len(letters) - len(digits)) > 1:
        return ""
    result = []
    if len(letters) > len(digits):
        for a, b in zip(letters, digits):
            result.append(a)
            result.append(b)
        result.append(letters[-1])
    else:
        for a, b in zip(digits, letters):
            result.append(a)
            result.append(b)
        if len(digits) > len(letters):
            result.append(digits[-1])
return ''.join(result)



86.def repeated_string(s, n):
return s.count('a') * (n // len(s)) + s[:n % len(s)].count('a')


87.def repeated_subarray_pattern(arr, m, k):
return arr[:m] * k == arr



88.def replace_elements_with_greatest_right(arr):
    max_right = -1
    for i in range(len(arr)-1, -1, -1):
        arr[i], max_right = max_right, max(max_right, arr[i])
return arr




89.def reverse_characters_in_words(s):
return ' '.join(word[::-1] for word in s.split())



90.def rob(nums):
    prev, curr = 0, 0
    for num in nums:
        prev, curr = curr, max(curr, prev + num)
return curr

91.def round_grades(grades):
    rounded = []
    for grade in grades:
        if grade < 38:
            rounded.append(grade)
        else:
            next_multiple = (grade // 5 + 1) * 5
            if next_multiple - grade < 3:
                rounded.append(next_multiple)
            else:
                rounded.append(grade)
return rounded



92.def shift_grid(grid, k):
    m, n = len(grid), len(grid[0])
    k %= m * n
    flat = [grid[i][j] for i in range(m) for j in range(n)]
    flat = flat[-k:] + flat[:-k]
return [flat[i*n:(i+1)*n] for i in range(m)]



93.from collections import Counter

def sort_by_frequency(nums: list[int]) -> list[int]:
    freq = Counter(nums)
    nums.sort(key=lambda x: (freq[x], -x))
    return nums


94.def sorted_squares(nums):
return sorted(x*x for x in nums)

95.from collections import Counter

def sum_of_unique_elements(nums: list[int]) -> int:
    freq = Counter(nums)
    return sum(num for num, count in freq.items() if count == 1)






96.def super_reduced_string(s):
    stack = []
    for c in s:
        if stack and stack[-1] == c:
            stack.pop()
        else:
            stack.append(c)
return ''.join(stack) if stack else "Empty String"



97.def tic_tac_toe_winner(moves: list[list[int]]) -> str:
    board = [[""] * 3 for _ in range(3)]
    for i, (r, c) in enumerate(moves):
        board[r][c] = "A" if i % 2 == 0 else "B"

    for player in ["A", "B"]:
       
        for i in range(3):
            if all(board[i][j] == player for j in range(3)) or \
               all(board[j][i] == player for j in range(3)):
                return player
       
        if all(board[i][i] == player for i in range(3)) or \
           all(board[i][2 - i] == player for i in range(3)):
            return player

    return "Draw" if len(moves) == 9 else "Pending"



98.def to_hexspeak(num):
    hex_str = hex(int(num))[2:].upper().replace('0', 'O').replace('1', 'I')
    if any(c not in 'ABCDEFIO' for c in hex_str):
        return "ERROR"
return hex_str




99.def unique_paths_with_obstacles(obstacle_grid):
    if not obstacle_grid or obstacle_grid[0][0] == 0:
        return 0

    m, n = len(obstacle_grid), len(obstacle_grid[0])
    dp = [[0]*n for _ in range(m)]
    dp[0][0] = 1  

    for i in range(m):
        for j in range(n):
            if obstacle_grid[i][j] == 0:
                dp[i][j] = 0
            else:
                if i > 0:
                    dp[i][j] += dp[i-1][j]
                if j > 0:
                    dp[i][j] += dp[i][j-1]

    return dp[m-1][n-1]



100.def unplaced_fruits(fruits: list[int], baskets: list[int]) -> int:
    fruits.sort(reverse=True)
    baskets.sort()
    used = [False] * len(baskets)
    unplaced = 0

    for fruit in fruits:
        placed = False
        for i in range(len(baskets)):
            if not used[i] and baskets[i] >= fruit:
                used[i] = True
                placed = True
                break
        if not placed:
            unplaced += 1

    return unplaced




101.def valid_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            # Try skipping either left or right character
            return s[left:right] == s[left:right][::-1] or s[left+1:right+1] == s[left+1:right+1][::-1]
        left += 1
        right -= 1
    return True
 
        
