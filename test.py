# from math import factorial
# def combine_number(n,k):
#     combinations = factorial(n) / factorial(n - k)
#     return combinations

# # 定义一个函数来计算给定字符串中平衡串的子序列数量
# def count_balanced_subsequences(s):
#     MOD = 10**9 + 7
#     # 初始化字母计数器
#     count = [0] * 26
#     for char in s:
#         count[ord(char) - ord('a')] += 1
    
#     result = 0
#     while sum(count)>0:
#         cnt = 0
#         for i in range(26):
#             if count[i] > 0:
#                 cnt += 1
#                 count[i] -= 1
#         print(count)
#         result += combine_number(cnt,2)
#         print(result)
#     return result

# # 示例输入
# n = 5
# s = "ababc"
# # 计算结果
# result = count_balanced_subsequences(s)
# print(result)

# def calculate_scores(k, x, y):
#     # 根据题目描述，我们有以下等式：
#     # a + b + c = k
#     # c = a + x
#     # c = b - y
#     # 由上面两个等式可得：a + x = b - y
#     # 因此我们可以解这个线性方程组得到a, b, c的值

#     # a + (a + x) + (a + x + y) = k
#     # 3a + 2x + y = k
#     # 解得：
#     a = (k - 2*x - y) // 3
#     c = a + x
#     b = c + y

#     return a, b, c

# # 示例输入
# k, x, y = 441, 1, -20
# # 计算输出
# a, b, c = calculate_scores(k, x, y)
# print(a, b, c)



# def min_operations_to_equal(s, t):
#     operations = []
#     i = len(s) - 1
#     while i >= 0:
#         if s[i] != t[i]:
#             operations.append((1, i + 1, t[i]))
#             s = t[i] * (i + 1) + s[i + 1:]  # 更新s字符串为操作后的状态
#         i -= 1
#     return operations

# # 示例输入
# s = "aabc"
# t = "abcc"

# # 计算所需的最小操作次数及具体操作
# operations1 = min_operations_to_equal(s, t)
# operations2 = min_operations_to_equal(t, s)

# if len(operations1) < len(operations2):
#     operations = operations1
# else:
#     operations = operations2
# # 输出结果
# print(len(operations))
# for op in operations:
#     print(*op)



# #include <iostream>
# #include <linux/limits,h>
# #include <string>
# using namespace std;
# struct node {
# char a, b;
# }ans[100100];

# char col[9] = {'','a','b'

# int main(){
#     string x; cin >>x;
#     char w=x[0],s=x[1];
#     int ww=w-'a'+ 1;
#     int ss =s -'0';
#     int tot = 0;
#     for(int i=1;i<=8; ++i){
#         if(1 != ss){
#             ans[++tot]=(node){w,(char)(i +'@')};
#         }
#     }
#     for(int i=1; i <= 8; ++i){
#         if(1 != ww){
#             ans[++tot]=(node){colli], s};
#         }
#     }

#     for(int i=1;i<8;++1){
#         for(int j=1;j<=8; ++j){
#             if(i+j==ww + ss){
#                 if(i !=ss |lj != ww){
#                     ans[++tot]=(node){col[j],(char)}
#                 }
#             }
#         }
#     } 
#     for(int i=1;i<=8;++i){
#         for(int j=1;j<=8; ++j){
#             if(i-j=ss-ww){
#                 if(i !=ss ||j!= ww){
#                     ans[++tot]=(node)fcol[j]，(char)
#                 }
#             }
#         }
#     }

#     cout << tot.<< endl;
#     for(int i =1; i <= tot; ++i) cout << ans[i].a <<
#     return 0;

# #include <linux/limits.h>
# #include <string>using namespace stdj
# const int N= 1001008
# int n, a[N];
# int main(){
#     cin >> n;
#     int tot1 = 0, tot0 = 0;
#     for(int i=1;i<= n; ++1){
#         cin >> a[i];
#         if(a[i]%2=0) tot0++;
#         else tot1++;
#     cout << abs(tot1-(n/2))<<endl;
#     return 0;
