import java.util.*;
import util.*;
public class LeetCode3 {
    class Solution93 {
        static final int SEG_COUNT = 4;
        List<String> ans = new ArrayList<String>();
        int[] segments = new int[SEG_COUNT];

        public List<String> restoreIpAddresses(String s) {
            segments = new int[SEG_COUNT];
            dfs(s, 0, 0);
            return ans;
        }

        public void dfs(String s, int segId, int segStart) {
            // 如果找到了 4 段 IP 地址并且遍历完了字符串，那么就是一种答案
            if (segId == SEG_COUNT) {
                if (segStart == s.length()) {
                    StringBuffer ipAddr = new StringBuffer();
                    for (int i = 0; i < SEG_COUNT; ++i) {
                        ipAddr.append(segments[i]);
                        if (i != SEG_COUNT - 1) {
                            ipAddr.append('.');
                        }
                    }
                    ans.add(ipAddr.toString());
                }
                return;
            }

            // 如果还没有找到 4 段 IP 地址就已经遍历完了字符串，那么提前回溯
            if (segStart == s.length()) {
                return;
            }

            // 由于不能有前导零，如果当前数字为 0，那么这一段 IP 地址只能为 0
            if (s.charAt(segStart) == '0') {
                segments[segId] = 0;
                dfs(s, segId + 1, segStart + 1);
            }

            // 一般情况，枚举每一种可能性并递归
            int addr = 0;
            for (int segEnd = segStart; segEnd < s.length(); ++segEnd) {
                addr = addr * 10 + (s.charAt(segEnd) - '0');
                if (addr > 0 && addr <= 0xFF) {
                    segments[segId] = addr;
                    dfs(s, segId + 1, segEnd + 1);
                } else {
                    break;
                }
            }
        }
    }

    //X包围O
    public class Solution130{
        int n,m;
        public void solve(char[][] board) {
            n = board.length;
            if (n == 0) {
                return;
            }
            m = board[0].length;
            for (int i = 0; i < n; i++) {
                dfs(board, i, 0);
                dfs(board, i, m - 1);
            }
            for (int i = 1; i < m - 1; i++) {
                dfs(board, 0, i);
                dfs(board, n - 1, i);
            }
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    if (board[i][j] == 'A') {
                        board[i][j] = 'O';
                    } else if (board[i][j] == 'O') {
                        board[i][j] = 'X';
                    }
                }
            }
        }
        public void dfs(char[][] board, int x, int y) {
            if (x < 0 || x >= n || y < 0 || y >= m || board[x][y] != 'O') {
                return;
            }
            board[x][y] = 'A';
            dfs(board, x + 1, y);
            dfs(board, x - 1, y);
            dfs(board, x, y + 1);
            dfs(board, x, y - 1);
        }
    }

    //深拷贝无向图
    class Node {
        public int val;
        public List<Node> neighbors;

        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }

        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }

        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }
    public class Solution133{
        private HashMap <Node, Node> visited = new HashMap <> ();
        public Node cloneGraph(Node node) {
            if (node == null) {
                return node;
            }

            // 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
            if (visited.containsKey(node)) {
                return visited.get(node);
            }

            // 克隆节点，注意到为了深拷贝我们不会克隆它的邻居的列表
            Node cloneNode = new Node(node.val, new ArrayList());
            // 哈希表存储
            visited.put(node, cloneNode);

            // 遍历该节点的邻居并更新克隆节点的邻居列表
            for (Node neighbor: node.neighbors) {
                cloneNode.neighbors.add(cloneGraph(neighbor));
            }
            return cloneNode;
        }
    }

    /**给出一些不同颜色的盒子，盒子的颜色由数字表示，即不同的数字表示不同的颜色。
     你将经过若干轮操作去去掉盒子，直到所有的盒子都去掉为止。每一轮你可以移除具有相同颜色的连续 k 个盒子（k >= 1），
     这样一轮之后你将得到 k*k 个积分。
     当你将所有盒子都去掉之后，求你能获得的最大积分和。*/
    public class Solution546 {

        public int removeBoxes(int[] boxes) {
            int[][][] dp = new int[100][100][100];
            return calculatePoints(boxes, dp, 0, boxes.length - 1, 0);
        }

        public int calculatePoints(int[] boxes, int[][][] dp, int l, int r, int k) {
            if (l > r) return 0;
            if (dp[l][r][k] != 0) return dp[l][r][k];
            while (r > l && boxes[r] == boxes[r - 1]) {
                r--;
                k++;
            }
            dp[l][r][k] = calculatePoints(boxes, dp, l, r - 1, 0) + (k + 1) * (k + 1);
            for (int i = l; i < r; i++) {
                if (boxes[i] == boxes[r]) {
                    dp[l][r][k] = Math.max(dp[l][r][k], calculatePoints(boxes, dp, l, i, k + 1) + calculatePoints(boxes, dp, i + 1, r - 1, 0));
                }
            }
            return dp[l][r][k];
        }
    }

    /**在 "100 game" 这个游戏中，两名玩家轮流选择从 1 到 10 的任意整数，累计整数和，先使得累计整数和达到 100 的玩家，即为胜者。

     如果我们将游戏规则改为 “玩家不能重复使用整数” 呢？

     例如，两个玩家可以轮流从公共整数池中抽取从 1 到 15 的整数（不放回），直到累计整数和 >= 100。

     给定一个整数 maxChoosableInteger （整数池中可选择的最大数）和另一个整数 desiredTotal（累计和），
     判断先出手的玩家是否能稳赢（假设两位玩家游戏时都表现最佳）？

     你可以假设 maxChoosableInteger 不会大于 20， desiredTotal 不会大于 300。*/
    /**
     * @Description:
     * 由于状态不可用数组进行传递【在递归当中会受到改变，不能准确定位当前状态】，故在此处用Int的位表示状态（1表示用过,0表示未用过）
     * 这里采用DP状态压缩的方式，思想与回溯类似，只是这里的状态被压缩成了一个bitArray了
     * 状态压缩，我们可以用二进制的第i位的0或者1来表示i这个数字的选取与否，这样所有数字的选取状态就可以用一个数来很方便的表示，
     * 题目说了不超过20位，所以这里就可以用一个int来表示状态state，通过state来判断状态是否一致，进而进行记忆化的存取
     */
    public class Solution464 {

        public boolean canIWin(int maxChoosableInteger, int desiredTotal) {

            if (maxChoosableInteger >= desiredTotal) return true;
            if ((1 + maxChoosableInteger) * maxChoosableInteger / 2 < desiredTotal) return false;
            /**
             *  dp表示"每个"取数(A和B共同作用的结果)状态下的输赢
             *  例如只有1,2两个数选择，那么 (1 << 2) - 1 = 4 - 1 = 3种状态表示：
             *  01,10,11分别表示：A和B已经选了1，已经选了2，已经选了1、2状态下，A的输赢情况
             *  并且可见这个表示所有状态的dp数组的每个状态元素的长度为maxChoosableInteger位的二进制数
             */
            Boolean[] dp = new Boolean[(1 << maxChoosableInteger) - 1];
            return dfs(maxChoosableInteger, desiredTotal, 0, dp);
        }

        /**
         * @param maxChoosableInteger 选择的数的范围[1,2,...maxChoosableInteger]
         * @param desiredTotal 目标和
         * @param state 当前状态的十进制表示（记录着可能不止一个数被选择的状态）
         * @param dp 记录所有状态
         * @return
         */
        private boolean dfs(int maxChoosableInteger, int desiredTotal, int state, Boolean[] dp) {
            if (dp[state] != null)
                return dp[state];
            /**
             * 例如maxChoosableInteger=2，选择的数只有1,2两个，二进制只要两位就可以表示他们的选择状态
             * 最大位为2（第2位），也就是1 << (2 - 1)的结果，所以最大的位可以表示为  1 << (maxChoosableInteger - 1)
             * 最小的位可以表示为 1 << (1 - 1)，也就是1（第1位）
             * 这里i表示括号的范围
             */
            for (int i = 1; i <= maxChoosableInteger; i++){
                //当前待抉择的位，这里的tmp十进制只有一位为1，用来判断其为1的位，对于state是否也是在该位上为1
                //用以表示该位（数字）是否被使用过
                /**
                 * (&运算规则，都1才为1)
                 * 例如,i=3, tmp = 4, state = 3;
                 *  100
                 * &011
                 * =0  表示该位没有被使用过，也就是第三位没有被使用过，即数字3 (i)没有被使用过
                 */
                int tmp = (1 << (i - 1));
                if ((tmp & state) == 0){  //该位没有被使用过
                    //如果当前选了i已经赢了或者选了i还没赢但是后面对方选择输了,tmp|state表示进行状态的更新
                    /**
                     * 例如
                     *  100
                     * |011
                     * =111
                     */
                    //注意这里并没有像回溯一样进行状态的(赋值化的)更新、回溯
                    //其实这里是隐含了回溯的，我们通过参数传递更新后的state
                    //但是我们在这个调用者这里的state还是没有进行更新的，所以
                    //就相当于回溯了状态。
                    if (desiredTotal - i <= 0 || !dfs(maxChoosableInteger, desiredTotal - i, tmp|state, dp)) {
                        dp[state] = true;
                        return true;
                    }
                }
            }
            //如果都赢不了
            dp[state] = false;
            return false;
        }
    }
    //判断二叉树高度是否平衡
    class Solution110 {
        public boolean isBalanced(TreeNode root) {
            return recur(root) != -1;
        }

        private int recur(TreeNode root) {
            if (root == null) return 0;
            int left = recur(root.left);
            if(left == -1) return -1;
            int right = recur(root.right);
            if(right == -1) return -1;
            return Math.abs(left - right) < 2 ? Math.max(left, right) + 1 : -1;
        }
    }
    //判断链表是不是二叉树的一条路径
    class Solution1367 {
        public boolean isSubPath(ListNode head, TreeNode root) {
            if (head == null) {
                return true;
            }
            if (root == null) {
                return false;
            }
            //先判断当前的节点，如果不对，再看左子树和右子树呗
            return isSub(head, root) || isSubPath(head, root.left) || isSubPath(head, root.right);
        }

        private boolean isSub(ListNode head, TreeNode node) {
            //特判：链表走完了，返回true
            if (head == null) {
                return true;
            }
            //特判：链表没走完，树走完了，这肯定不行，返回false
            if (node == null) {
                return false;
            }
            //如果值不同，必定不是啊
            if (head.val != node.val) {
                return false;
            }
            //如果值相同，继续看，左边和右边有一个满足即可
            return isSub(head.next, node.left) || isSub(head.next, node.right);
        }
    }
    //排序链表转二叉搜索树
    class Solution109 {
        ListNode globalHead;
        public TreeNode sortedListToBST(ListNode head) {
            globalHead=head;
            int length=getLength(head);
            return buildTree(0,length-1);
        }

        public int getLength(ListNode head){
            int cnt=0;
            while(head!=null){
                cnt++;
                head=head.next;
            }
            return cnt;
        }

        public TreeNode buildTree(int left,int right){
            if(left>right)
                return null;
            int mid=(left+right+1)/2;
            TreeNode root=new TreeNode(0);
            root.left=buildTree(left,mid-1);
            root.val=globalHead.val;
            globalHead=globalHead.next;
            root.right=buildTree(mid+1,right);
            return root;
        }
    }
    //机器人走路，只能向下或者向右
    class Solution62 {
        public int uniquePaths(int m, int n) {
            if(m<=0||n<=0){
                return 0;
            }
            int[][] dp=new int[m][n];
            for(int i=0;i<m;i++){
                dp[i][0]=1;
            }
            for(int i=0;i<n;i++){
                dp[0][i]=1;
            }
            for(int i=1;i<m;i++){
                for(int j=1;j<n;j++){
                    dp[i][j]=dp[i-1][j]+dp[i][j-1];
                }
            }
            return dp[m-1][n-1];
        }
    }
    //汉诺塔问题
    class SolutionM0806 {
        public void hanota(List<Integer> A, List<Integer> B, List<Integer> C) {
            hanoi(A.size(), A, B, C);
        }

        public void hanoi(int n, List<Integer> A, List<Integer> B, List<Integer> C){

            if(n == 1){
                C.add(A.get(A.size() - 1));
                A.remove(A.size() - 1);
            }else{
                //把A经过辅助C放到B上
                hanoi(n - 1, A, C, B);
                //把A放到C上
                C.add(A.get(A.size() - 1));
                A.remove(A.size() - 1);
                //把B经过辅助A放到C上
                hanoi(n - 1, B, A, C);
            }
        }
    }

    /**最长连续1*/
    public class Solution1004{
        public int longestOnes(int[] A, int K) {
            int left=0,right=0,cnt=0;
            for(;right<A.length;right++){
                if(A[right]==0)
                    cnt++;
                if(cnt>K){
                    if(A[left]==0)
                        cnt--;
                    left++;
                }
            }
            //最后right多加了1所以要减去
            right--;
            return right-left+1;
        }
    }

    /**移除K个数字使得结果最大*/
    class Solution402 {
        public String removeKdigits(String num, int k) {
            LinkedList<Character> stack = new LinkedList<Character>();

            for(char digit : num.toCharArray()) {
                while(stack.size() > 0 && k > 0 && stack.peekLast() > digit) {
                    stack.removeLast();
                    k -= 1;
                }
                stack.addLast(digit);
            }

            /* remove the remaining digits from the tail. */
            for(int i=0; i<k; ++i) {
                stack.removeLast();
            }

            // build the final string, while removing the leading zeros.
            StringBuilder ret = new StringBuilder();
            boolean leadingZero = true;
            for(char digit: stack) {
                if(leadingZero && digit == '0') continue;
                leadingZero = false;
                ret.append(digit);
            }

            /* return the final string  */
            if (ret.length() == 0) return "0";
            return ret.toString();
        }
    }
    /**最长重复字母子串*/
    public class Solution424{
        private int[] map = new int[26];

        public int characterReplacement(String s, int k) {
            if (s == null) {
                return 0;
            }
            char[] chars = s.toCharArray();
            int left = 0;
            int right = 0;
            int historyCharMax = 0;
            for(right=0;right<chars.length;right++){
                int index=chars[right]-'A';
                map[index]++;
                historyCharMax=Math.max(historyCharMax,map[index]);
                if(right-left+1>historyCharMax+k){
                    map[chars[left]-'A']--;
                    left++;
                }
            }
            //最后只要输出窗口长度就行
            return chars.length-left;
        }
    }
    /**前K个高频元素*/
    public class Solution347{
        public List<Integer> topKFrequent(int[] nums, int k) {
            HashMap<Integer,Integer> map=new HashMap<>();
            for(int num:nums){
                if(map.containsKey(num)){
                    map.put(num,map.get(num)+1);
                }else{
                    map.put(num,1);
                }
            }
            //小顶堆
            PriorityQueue<Integer> pq=new PriorityQueue<>(new Comparator<Integer>() {
                @Override
                public int compare(Integer o1, Integer o2) {
                    return map.get(o1)-map.get(o2);
                }
            });
            for(Integer key:map.keySet()){
                if(pq.size()<k){
                    pq.offer(key);
                }else if(map.get(key)>map.get(pq.peek())){
                    pq.poll();
                    pq.offer(key);
                }
            }
            List<Integer> ret=new LinkedList<>();
            while(!pq.isEmpty()){
                ret.add(pq.poll());
            }
            return ret;
        }
    }
    /**跳到末尾的最小步数*/
    public class Solution45{
        public int jump(int[] nums) {
            int step=0;
            int index=0;
            int maxIndex=nums.length-1;
            while(index<maxIndex){
                step++;
                //先看看最远能不能超过边界
                int nextIndex=index+nums[index];
                if(nextIndex>=maxIndex)
                    break;
                //不能到边界，则找范围内最远的点
                nextIndex=index+1;
                for(int i=1;i<=nums[index];i++){
                    if(index+i+nums[index+i]>nextIndex+nums[nextIndex]){
                        nextIndex=index+i;
                    }
                }
                index=nextIndex;
            }
            return step;
        }
    }
    /**数组中的不重复数字的全排列*/
    public class Solution46{
        public List<List<Integer>> permute(int[] nums) {
            List<List<Integer>> res=new LinkedList<>();
            int len=nums.length;
            if(len<=0||nums==null){
                return res;
            }
            boolean[] used=new boolean[len];
            List<Integer> path=new LinkedList<>();
            dfs(nums,used,0,len,path,res);
            return res;
        }

        public void dfs(int[] nums,boolean[] used,int depth,int len,List<Integer> path,List<List<Integer>> res){
            if(depth==len){
                res.add(new LinkedList<>(path));
                return;
            }
            for(int i=0;i<len;i++){
                if(!used[i]){
                    path.add(nums[i]);
                    used[i]=true;
                    dfs(nums,used,depth+1,len,path,res);
                    used[i]=false;
                    path.remove(path.size()-1);
                }
            }
        }
    }
    /**组合总和*/
    public class Solution39{
        public List<List<Integer>> combinationSum(int[] candidates, int target){
            List<List<Integer>> ret=new LinkedList<>();
            if(candidates.length<=0||candidates==null){
                return ret;
            }
            Arrays.sort(candidates);
            List<Integer> path=new LinkedList<>();
            dfs(candidates,0,candidates.length,target,path,ret);
            return ret;
        }

        public void dfs(int[] candidates,int begin,int len,int target,List<Integer> path,List<List<Integer>> ret){
            if(target==0){
                ret.add(new LinkedList<Integer>(path));
                return;
            }
            for(int i=begin;i<len;i++){
                if(target-candidates[i]<0)
                    break;
                path.add(candidates[i]);
                dfs(candidates,i,len,target-candidates[i],path,ret);
                path.remove(path.size()-1);
            }
        }
    }
    /**组合求和2*/
    class Solution40 {
        public List<List<Integer>> combinationSum2(int[] candidates, int target) {
            List<List<Integer>> ret=new LinkedList<>();
            if(candidates.length<=0||candidates==null){
                return ret;
            }
            Arrays.sort(candidates);
            List<Integer> path=new LinkedList<>();
            dfs(candidates,target,0,candidates.length,path,ret);
            return ret;
        }
        public void dfs(int[] candidates,int target,int begin,int len,List<Integer> path,List<List<Integer>> ret){
            if(target==0){
                ret.add(new LinkedList<>(path));
                return;
            }
            for(int i=begin;i<len;i++){
                if(target-candidates[i]<0){
                    break;
                }
                if(i>begin&&candidates[i]==candidates[i-1]){
                    continue;
                }
                path.add(candidates[i]);
                dfs(candidates,target-candidates[i],i+1,len,path,ret);
                path.remove(path.size()-1);
            }
        }
    }
    /**打家劫舍*/
    public class Solution198{
        public int rob(int[] nums) {
            if(nums.length<=0||nums==null){
                return 0;
            }
            int[] dp=new int[nums.length];
            dp[0]=nums[0];
            if(nums.length==1)
                return dp[0];
            dp[1]=Math.max(nums[0],nums[1]);
            for(int i=2;i<nums.length;i++){
                dp[i]=Math.max(dp[i-1],dp[i-2]+nums[i]);
            }
            return dp[nums.length-1];
        }
    }
    /**滑动窗口持续最大值*/
    class Solution239 {
        ArrayDeque<Integer> dq=new ArrayDeque<>();
        public int[] maxSlidingWindow(int[] nums, int k) {
            int n=nums.length;
            int maxidx=0;
            if(n*k==0){
                return new int[0];
            }
            if(k==1)
                return nums;
            for(int i=0;i<k;i++){
                //cleanDequeue(i,k,nums);
                dq.addLast(i);
                if(nums[i]>nums[maxidx])
                    maxidx=i;
            }
            int[] ret=new int[n-k+1];
            ret[0]=nums[maxidx];
            for(int i=k;i<n;i++){
                cleanDequeue(i,k,nums);
                dq.addLast(i);
                ret[i-k+1]=nums[dq.getFirst()];
            }
            return ret;
        }
        public void cleanDequeue(int i,int k,int[] nums){
            // remove indexes of elements not from sliding window
            if (!dq.isEmpty() && dq.getFirst() == i - k)
                dq.removeFirst();

            // remove from deq indexes of all elements
            // which are smaller than current element nums[i]
            while (!dq.isEmpty() && nums[i] > nums[dq.getLast()])
                dq.removeLast();
        }
    }
    /**组合求和3*/
    class Solution216 {
        public List<List<Integer>> combinationSum3(int k, int n) {
            List<List<Integer>> ret=new LinkedList<>();
            if(n>45){
                return ret;
            }
            boolean[] used=new boolean[9];
            List<Integer> path=new LinkedList<>();
            //不能使用重复数字就用used数组，顺序颠倒算同一组就用begin，因此这里都用上了
            dfs(k,n,1,used,path,ret);
            return ret;
        }
        public void dfs(int k,int n,int begin,boolean[] used,List<Integer> path,List<List<Integer>> ret){
            if(k==0&&n==0){
                ret.add(new LinkedList<>(path));
                return;
            }
            if(k==0)
                return;
            for(int i=begin;i<10;i++){
                if(!used[i-1]){
                    used[i-1]=true;
                    path.add(i);
                    dfs(k-1,n-i,i+1,used,path,ret);
                    used[i-1]=false;
                    path.remove(path.size()-1);
                }
            }
        }
    }
    /**n皇后问题*/
    class Solution51 {
        private int n;
        // 记录某一列是否放置了皇后
        private boolean[] col;
        // 记录主对角线上的单元格是否放置了皇后
        private boolean[] main;
        // 记录了副对角线上的单元格是否放置了皇后
        private boolean[] sub;
        private List<List<String>> res;

        public List<List<String>> solveNQueens(int n) {
            res = new ArrayList<>();
            if (n == 0) {
                return res;
            }

            // 设置成员变量，减少参数传递，具体作为方法参数还是作为成员变量，请参考团队开发规范
            this.n = n;
            this.col = new boolean[n];
            this.sub = new boolean[2 * n - 1];
            this.main = new boolean[2 * n - 1];
            Deque<Integer> path = new ArrayDeque<>();
            dfs(0, path);
            return res;
        }

        private void dfs(int row, Deque<Integer> path) {
            if (row == n) {
                // 深度优先遍历到下标为 n，表示 [0.. n - 1] 已经填完，得到了一个结果
                List<String> board = convert2board(path);
                res.add(board);
                return;
            }

            // 针对下标为 row 的每一列，尝试是否可以放置
            for (int j = 0; j < n; j++) {
                if (!col[j] && !sub[row + j] && !main[row - j + n - 1]) {
                    path.addLast(j);
                    col[j] = true;
                    //行下标与列下标之和表示一条线
                    sub[row + j] = true;
                    //行下标与列下标之差表示一条线，起点在main[n-1]，防止下标为负
                    main[row - j + n - 1] = true;

                    dfs(row + 1, path);

                    main[row - j + n - 1] = false;
                    sub[row + j] = false;
                    col[j] = false;
                    path.removeLast();
                }
            }
        }

        private List<String> convert2board(Deque<Integer> path) {
            List<String> board = new ArrayList<>();
            for (Integer num : path) {
                StringBuilder row = new StringBuilder();
                row.append(".".repeat(Math.max(0, n)));
                row.replace(num, num + 1, "Q");
                board.add(row.toString());
            }
            return board;
        }
    }
    /**二叉树层的均值*/
    class Solution637 {
        List<Double> res=new LinkedList<>();
        public List<Double> averageOfLevels(TreeNode root) {
            if(root==null)
                return res;
            Queue<TreeNode> q=new LinkedList<>();
            q.offer(root);
            helper(q);
            return res;
        }
        public void helper(Queue<TreeNode> q){
            if(q.isEmpty())
                return;
            long sum=0;
            int cnt=0;
            Queue<TreeNode> q2=new LinkedList<>();
            while(!q.isEmpty()){
                TreeNode tmp=q.poll();
                sum+=tmp.val;
                cnt++;
                if(tmp.left!=null)
                    q2.offer(tmp.left);
                if(tmp.right!=null)
                    q2.offer(tmp.right);
            }
            res.add((double)sum/cnt);
            helper(q2);
        }
    }
    /**单词搜索*/
    class Solution79 {
        private boolean[][] marked;

        //        x-1,y
        // x,y-1  x,y    x,y+1
        //        x+1,y
        private int[][] direction = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
        // 盘面上有多少行
        private int m;
        // 盘面上有多少列
        private int n;
        private String word;
        private char[][] board;

        public boolean exist(char[][] board, String word) {
            m = board.length;
            if (m == 0) {
                return false;
            }
            n = board[0].length;
            marked = new boolean[m][n];
            this.word = word;
            this.board = board;

            for(int i=0;i<m;i++){
                for(int j=0;j<n;j++){
                    if(dfs(i,j,0))
                        return true;
                }
            }
            return false;
        }

        private boolean dfs(int i, int j, int start) {
            if (start == word.length() - 1) {
                return board[i][j] == word.charAt(start);
            }
            if(board[i][j]==word.charAt(start)){
                marked[i][j]=true;
                for(int k=0;k<4;k++){
                    int newX=i+direction[k][0];
                    int newY=j+direction[k][1];
                    if(inArea(newX,newY)&&!marked[newX][newY]){
                        if(dfs(newX,newY,start+1))
                            return true;
                    }
                }
                marked[i][j]=false;
            }
            return false;
        }

        private boolean inArea(int x, int y) {
            return x >= 0 && x < m && y >= 0 && y < n;
        }
    }
    /**找池塘大小*/
    class SolutionMian1619 {
        public int[] pondSizes(int[][] land) {
            List<Integer> list = new ArrayList<>();
            int temp;

            // 遍历矩阵每个元素
            for (int i = 0; i < land.length; i++) {
                for (int j = 0; j < land[0].length; j++) {
                    temp = findPool(land, i, j);
                    if (temp != 0) list.add(temp);
                }
            }

            // 第一种List<Integer>转int[]
            // int[] result = new int[list.size()];
            // for (int i = 0; i < result.length; i++) {
            //   result[i] = list.get(i);
            // }

            // 第二种List<Integer>转int[]，优雅且高效
            int[] result = list.stream().mapToInt(Integer::valueOf).toArray();

            Arrays.sort(result);

            return result;
        }
        public int findPool(int[][] land,int x,int y){
            int num=0;
            int row=land.length;
            int col=land[0].length;
            if(x<0||x>row-1||y<0||y>col-1||land[x][y]!=0){
                return num;
            }
            num++;
            land[x][y]=-1;
            num+=findPool(land,x-1,y);
            num+=findPool(land,x+1,y);
            num+=findPool(land,x,y+1);
            num+=findPool(land,x,y-1);
            num+=findPool(land,x-1,y-1);
            num+=findPool(land,x-1,y+1);
            num+=findPool(land,x+1,y-1);
            num+=findPool(land,x+1,y+1);
            return num;
        }
    }
    /**最长递增子序列的个数*/
    class Solution673 {
        //动态规划
        public int findNumberOfLIS(int[] nums) {
            int N = nums.length;
            if (N <= 1) return N;
            int[] lengths = new int[N]; //lengths[i] = length of longest ending in nums[i]
            int[] counts = new int[N]; //count[i] = number of longest ending in nums[i]
            Arrays.fill(counts, 1);

            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < j; ++i) {
                    if(nums[i]<nums[j]){
                        if(lengths[i]>=lengths[j]){
                            lengths[j]=lengths[i]+1;
                            counts[j]=counts[i];
                        }else if(lengths[i]+1==lengths[j]){
                            counts[j]+=counts[i];
                        }
                    }
                }
            }

            int longest = 0, ans = 0;
            for (int length: lengths) {
                longest = Math.max(longest, length);
            }
            for (int i = 0; i < N; ++i) {
                if (lengths[i] == longest) {
                    ans += counts[i];
                }
            }
            return ans;
        }
    }
    /**岛屿数量*/
    public class Solution200{
        public int numIslands(char[][] grid) {
            if(grid==null||grid.length==0){
                return 0;
            }
            int row=grid.length;
            int col=grid[0].length;
            int cnt=0;
            for(int i=0;i<row;i++){
                for(int j=0;j<col;j++){
                    if(grid[i][j]=='1'){
                        cnt++;
                        dfs(grid,i,j);
                    }
                }
            }
            return cnt;
        }
        public void dfs(char[][] grid,int x,int y){
            int row=grid.length;
            int col=grid[0].length;
            if(x<0||x>=row||y<0||y>=col||grid[x][y]!='1'){
                return;
            }
            grid[x][y]='0';
            dfs(grid,x-1,y);
            dfs(grid,x+1,y);
            dfs(grid,x,y+1);
            dfs(grid,x,y-1);
        }
    }
    /**扫雷*/
    class Solution529 {
        public char[][] updateBoard(char[][] board, int[] click) {
            int x=click[0],y=click[1];
            int row=board.length;
            int col=board[0].length;
            dfs(board,x,y,row,col);
            return board;
        }
        //true就是游戏结束
        public void dfs(char[][] board,int x,int y,int row,int col){
            if(x<0||x>=row||y<0||y>=col)
                return;
            if(board[x][y]=='B')
                return;
            if(board[x][y]=='E'){
                int num=searchBomb(board,x,y,row,col);
                if(num>0){
                    board[x][y]=(char)('0'+num);
                    return;
                }else{
                    board[x][y]='B';
                    dfs(board,x-1,y,row,col);
                    dfs(board,x+1,y,row,col);
                    dfs(board,x,y-1,row,col);
                    dfs(board,x,y+1,row,col);
                    dfs(board,x-1,y-1,row,col);
                    dfs(board,x-1,y+1,row,col);
                    dfs(board,x+1,y-1,row,col);
                    dfs(board,x+1,y+1,row,col);
                }
            }
            if(board[x][y]=='M'){
                board[x][y]='X';
                return;
            }
        }
        public int searchBomb(char[][] board,int x,int y,int row,int col){
            int num=0;
            if(inArea(x-1,y,row,col)){
                if(board[x-1][y]=='M')
                    num++;
            }
            if(inArea(x+1,y,row,col)){
                if(board[x+1][y]=='M')
                    num++;
            }
            if(inArea(x,y-1,row,col)){
                if(board[x][y-1]=='M')
                    num++;
            }
            if(inArea(x,y+1,row,col)){
                if(board[x][y+1]=='M')
                    num++;
            }
            if(inArea(x-1,y-1,row,col)){
                if(board[x-1][y-1]=='M')
                    num++;
            }
            if(inArea(x+1,y-1,row,col)){
                if(board[x+1][y-1]=='M')
                    num++;
            }
            if(inArea(x-1,y+1,row,col)){
                if(board[x-1][y+1]=='M')
                    num++;
            }
            if(inArea(x+1,y+1,row,col)){
                if(board[x+1][y+1]=='M')
                    num++;
            }
            return num;
        }
        public boolean inArea(int x,int y,int row,int col){
            return x>=0&&x<row&&y>=0&&y<col;
        }
    }

}
