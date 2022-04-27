# x = [[10, 20], [2, 3]]
#
# fig = plt.figure(figsize=(8, 5))
# plt.grid(visible=True)
# fig.suptitle('Test title', fontsize=10, x=0.5, y=0.05)
# fig.subplots_adjust(top=0.95, bottom=0.15)
# plt.plot(x, marker='o')
# plt.show()

# x = np.array([1, 7, 10, 3, 4, -10, 1, -3, -5, -9])
# print(x)
# print(x[np.argsort(x)[::-1]])
# print(x[select_k_best_abs(4)(x)])

#
# data = [[1, 2, 3], [4, 5, 6]]
# df = pd.DataFrame(data, columns=['x', 'y', 'z'])
# print(df)
# x = df['x']
# y = df['y']
# z = df['z']
# print(x, y, z)
# df['a'] = (y * x) * (z ** 2)
# print(df)
#
# df = pd.DataFrame([(.21, .32, 'str1'), (.01, .67, 'str2'), (.66, .03, 'str3'), (.21, .18, 'str4')],
#                   columns=['dogs', 'cats', 'strs'])
# df = df.round(1)
# data = [(1, float('inf')), (float('inf'), 2)]
# df = pd.DataFrame(data)
# print(df)
# print(df.replace([np.inf, -np.inf], 1))
# print(float('inf') - float('inf'))
# tt = [(1, 2), (2, 3)]
#
# df = pd.DataFrame(tt, columns=['x', 'y', 'z', 'a'])
# print(df)

# import seaborn as sns
#
# x = [[1, 2, 4], [2, 3, 4], [1, 3, 4], [2, 4, 4], [1, 1, 5], [2, 2, 5], [1, 4, 5], [2, 5, 5]]
#
# df = pd.DataFrame(data=x, columns=['x', 'y', 'model'])
# sns.lineplot(x='x', y='y', hue='model',
#              data=df, ci='sd', palette=['red', 'blue'])
# plt.grid()
#
# plt.show()


x = 'heh/kek/4.csv'
print(x[:-4][-1:])
