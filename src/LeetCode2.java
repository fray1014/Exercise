import org.junit.Test;
import java.util.*;

public class LeetCode2 {
    public static void main(String[] args){
        Thread t = new Thread() {
            public void run() {
                print();
            }
        };
        t.run();
        System.out.print("MT");
    }
    static void print() {
        System.out.print("DP");

    }
    /**深拷贝复杂链表*/
    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }
    public class SolutionJZ25 {
        public RandomListNode Clone(RandomListNode pHead){
            HashMap<RandomListNode,RandomListNode> hm=new HashMap<>();
            RandomListNode p=pHead;
            while(p!=null){
                RandomListNode tmp=new RandomListNode(p.label);
                hm.put(p,tmp);
                p=p.next;
            }
            p=pHead;
            while(p!=null){
                RandomListNode tmp=hm.get(p);
                tmp.next=p.next==null?null:hm.get(p.next);
                tmp.random=p.random==null?null:hm.get(p.random);
                p=p.next;
            }
            return hm.get(pHead);
        }
    }
    /**输入一颗二叉树的根节点和一个整数，按字典序打印出二叉树中结点值的和为输入整数的所有路径。
     * 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。*/
    public class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;

        }
    }
    public class SolutionJZ24 {
        public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {

            ArrayList<ArrayList<Integer>> result = new ArrayList<>();
            if (root == null){
                return result;
            }


            ArrayList<Integer> path = new ArrayList<>();
            this.find(root, target, result, path);
            return result;
        }


        private void find(TreeNode root, int target, ArrayList<ArrayList<Integer>> result, ArrayList<Integer> path) {
            // 0，当节点为空，return
            if (root == null) {
                return;
            }

            path.add(root.val);
            target -= root.val;

            // 1，当目标值小于0，return
            if(target < 0){
                return;
            }

            // 2，当目标值为0 并且 节点下无其他节点, 保存并返回
            if(target == 0 && root.left == null && root.right == null){
                result.add(path);
                return;
            }

            // 继续遍历左右节点
            // 这里new path是因为左右都会在下次递归path.add(root.val);
            this.find(root.left, target, result, new ArrayList<>(path));
            this.find(root.right, target, result, new ArrayList<>(path));
        }
    }

    /**
     牛牛有一个没有重复元素的数组a，他想要将数组内第n大的数字和第m大的数(从大到小排序)交换位置你能帮帮他吗。
     给定一个数组a，求交换第n大和第m大元素后的数组。*/
    public class Solution0730A{
        public int[] sovle (int[] a, int n, int m) {
            if(n==m||a.length==1)
                return a;
            if(a.length==0||a.length<n||a.length<m)
                return a;
            if(a.length==2){
                int swap=a[0];
                a[0]=a[1];
                a[1]=swap;
            }
            TreeSet<Integer> t=new TreeSet<>();
            for(int i:a){
                t.add(i);
            }
            int index=0;
            int[] b=new int[a.length];
            while(!t.isEmpty()){
                b[index++]=t.last();
                t.pollLast();
            }
            index=0;
            for(int i=0;i<a.length;i++){
                if(index==2)
                    break;
                if(a[i]==b[n-1]){
                    a[i]=b[m-1];
                    index++;
                }
                if(a[i]==b[m-1]){
                    a[i]=b[n-1];
                    index++;
                }
            }
            return a;
        }

    }
    /**输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
     * 如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。*/
    public class SolutionJZ23 {
        public boolean VerifySquenceOfBST(int [] sequence) {
            if(sequence.length==0)
                return false;
            return helpVerify(sequence,0,sequence.length-1);
        }

        public boolean helpVerify(int[] a,int start,int root){
            if(start>=root)
                return true;
            int i;
            int key=a[root];
            for(i=start;i<root;i++){
                if(a[i]>key)
                    break;
            }
            for(int j=i;j<root;j++){
                if(a[j]<key)
                    return false;
            }
            return helpVerify(a,start,i-1)&&helpVerify(a,i,root-1);
        }
    }

    /**输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序*/
    public static class SolutionJZ21{
        public boolean IsPopOrder(int [] pushA,int [] popA) {
            if(pushA.length!=popA.length||pushA.length==0||popA.length==0){
                return false;
            }
            Stack<Integer> s=new Stack<>();
            int indexi=0;
            int indexj=0;
            while(indexi<pushA.length){
                if(pushA[indexi]==popA[indexj]){
                    indexi++;
                    indexj++;
                }else{
                    s.push(pushA[indexi++]);
                }
            }
            while(!s.isEmpty()){
                if(s.pop()!=popA[indexj]){
                    return false;
                }
                indexj++;
            }
            return true;

        }
    }

    /**定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。*/
    public static class SolutionJZ20{
        static Stack<Integer> normal=new Stack<Integer>();
        static Stack<Integer> find=new Stack<Integer>();
        public void push(int node) {
            normal.push(node);
            if(find.empty()||find.peek()>=node){
                find.push(node);
            }
        }

        public void pop() {
            int tmp=normal.pop();
            if(tmp==find.peek()){
                find.pop();
            }
        }

        public int top() {
            return normal.peek();
        }

        public int min() {
            return find.peek();
        }
    }
    /**从上往下打印出二叉树的每个节点，同层节点从左至右打印。*/
    public static class SolutionJZ22 {

        public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
            Queue<TreeNode> queue=new LinkedList<>();
            ArrayList<Integer> arr=new ArrayList<>();
            if(root==null)
                return arr;
            queue.offer(root);
            while(!queue.isEmpty()){
                TreeNode tmp=queue.poll();
                arr.add(tmp.val);
                if(tmp.left!=null)
                    queue.offer(tmp.left);
                if(tmp.right!=null)
                    queue.offer(tmp.right);
            }
            return arr;
        }
    }

    /**输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字*/
    public static class SolutionJZ19{
        public ArrayList<Integer> printMatrix(int [][] matrix) {
            ArrayList<Integer> list=new ArrayList<>();
            if(matrix.length==0)
                return list;
            int up = 0;
            int down = matrix.length-1;
            int left = 0;
            int right = matrix[0].length-1;
            while(true){
                // 最上面一行
                for(int col=left;col<=right;col++){
                    list.add(matrix[up][col]);
                }
                // 向下逼近
                up++;
                // 判断是否越界
                if(up > down){
                    break;
                }
                // 最右边一行
                for(int row=up;row<=down;row++){
                    list.add(matrix[row][right]);
                }
                // 向左逼近
                right--;
                // 判断是否越界
                if(left > right){
                    break;
                }
                // 最下面一行
                for(int col=right;col>=left;col--){
                    list.add(matrix[down][col]);
                }
                // 向上逼近
                down--;
                // 判断是否越界
                if(up > down){
                    break;
                }
                // 最左边一行
                for(int row=down;row>=up;row--){
                    list.add(matrix[row][left]);
                }
                // 向右逼近
                left++;
                // 判断是否越界
                if(left > right){
                    break;
                }
            }
            return list;
        }
    }

    /**输入两棵二叉树A，B，判断B是不是A的子结构*/
    public static class SolutionJZ17{
        public boolean HasSubtree(TreeNode root1,TreeNode root2) {
            if(root2==null||root1==null)
                return false;
            return HasSubtree(root1.left,root2)||HasSubtree(root1.right,root2)||isNodeSame(root1,root2);
        }

        public boolean isNodeSame(TreeNode root1,TreeNode root2){
            if(root2==null)
                return true;
            if(root1==null)
                return false;
            if(root1.val!=root2.val){
                return false;
            }
            return isNodeSame(root1.left,root2.left)&&isNodeSame(root1.right,root2.right);

        }
    }

    /**反转链表*/
    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }
    public class SolutionJZ15 {
        public ListNode ReverseList(ListNode head) {
            ListNode pre=null;
            while(head!=null){
                ListNode next=head.next;
                head.next=pre;
                pre=head;
                head=next;
            }
            return pre;
        }
    }

    /**合并链表*/
    public class SolutionJZ16{
        public ListNode Merge(ListNode list1,ListNode list2) {
            ListNode head=new ListNode(1);
            ListNode cur=head;
            while(list1!=null&&list2!=null){
                if(list1.val<=list2.val){
                    cur.next=list1;
                    list1=list1.next;
                }else{
                    cur.next=list2;
                    list2=list2.next;
                }
                cur=cur.next;
            }
            if(list1!=null)
                cur.next=list1;
            if(list2!=null)
                cur.next=list2;
            return head.next;
        }
    }

    /**倒数第K个结点*/
    public class SolutionJZ14{
        public ListNode FindKthToTail(ListNode head,int k) {
            if(head==null||k==0)
                return head;
            ListNode slow=head;
            ListNode fast=head;
            for(int i=0;i<k;i++){
                if(fast==null)
                    return fast;
                fast=fast.next;
            }
            while(fast!=null){
                fast=fast.next;
                slow=slow.next;
            }
            return slow;
        }
    }

    /**输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，
     * 所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。*/
    public class SolutionJZ13 {
        public void reOrderArray(int [] array) {
            if(array.length==0||array==null){
                return;
            }
            int m=0;
            for(int i=0;i<array.length;i++){
                if(array[i]%2==1){
                    int j=i;
                    int tmp=array[i];
                    while(j>m){
                        array[j]=array[j-1];
                        j--;
                    }
                    m=j+1;
                    array[j]=tmp;
                }
            }
        }
    }

    /**算乘方，快速幂*/
    public class SolutionJZ12 {
        public double Power(double base, int exponent) {
            if(exponent<0){
                base=1/base;
                exponent=-exponent;
            }
            return helpPow(base,exponent);
        }
        public double helpPow(double base,int exponent){
            double tmp=1;
            if(exponent==0)
                return 1;
            if(exponent%2==1){
                exponent--;
                tmp=base;
            }
            while(exponent>1){
                base*=base;
                exponent/=2;
            }
            return base*tmp;
        }
    }

    /**统计二进制1的位数*/
    public class SolutionJZ11 {
        public int NumberOf1(int n) {
            int cnt=0;
            while(n!=0){
                cnt++;
                n=n&(n-1);
            }
            return cnt;
        }
    }

    /**矩形覆盖*/
    public class SolutionJZ10{
        //斐波那契数列
        public int RectCover(int target) {
            if(target<=0)
                return 0;
            if(target==1)
                return 1;
            if(target==2)
                return 2;
            return RectCover(target-1)+RectCover(target-2);
        }
    }

    /**一只青蛙一次可以跳上1级台阶，也可以跳上2级。
     * 求该青蛙跳上一个n级的台阶总共有多少种跳法*/
    public class SolutionJZ8 {
        public int JumpFloor(int target) {
            //斐波那契最优解，时间复杂度O(n)，空间复杂度O(1)
            if(target<=1)
                return 1;
            if(target==2)
                return 2;
            int target_1=1;
            int target_2=1;
            int ret=0;
            for(int i=2;i<=target;i++){
                ret=target_1+target_2;
                target_2=target_1;
                target_1=ret;
            }
            return ret;
        }
    }

    /**把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
     输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。*/
    public class SolutionJZ6 {
        public int minNumberInRotateArray(int [] array) {
            if(array.length==0||array==null)
                return 0;
            int low=0;
            int high=array.length-1;
            int mid;
            while(low<high){
                if(array[low]<array[high])
                    return array[low];
                mid=low+((high-low)>>1);
                if(array[mid]>array[high]){
                    low=mid+1;
                }else if(array[mid]<array[high]){
                    high=mid;
                }else{
                    high--;
                }

            }
            return array[low];
        }
    }

    /**输入前序遍历和中序遍历，还原二叉树*/
    public class SolutionJZ4 {
        public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
            if(pre.length==0||in.length==0)
                return null;
            TreeNode root=new TreeNode(pre[0]);
            for(int i=0;i<in.length;i++){
                if(in[i]==root.val){
                    root.left=reConstructBinaryTree(Arrays.copyOfRange(pre,1,i+1),
                            Arrays.copyOfRange(in,0,i));
                    root.right=reConstructBinaryTree(Arrays.copyOfRange(pre,i+1,pre.length),
                            Arrays.copyOfRange(in,i+1,in.length));
                }
            }
            return root;
        }
    }

    /**输入一个链表，按链表从尾到头的顺序返回一个ArrayList。*/
    public class SolutionJZ3{
        ArrayList<Integer> list = new ArrayList();
        public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
            if(listNode!=null){
                printListFromTailToHead(listNode.next);
                list.add(listNode.val);
            }
            return list;
        }
        /*
        public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
            ArrayList<Integer> list = new ArrayList<>();
            ListNode tmp = listNode;
            while(tmp!=null){
                list.add(0,tmp.val);
                tmp = tmp.next;
            }
            return list;
        }*/
    }
    /**请实现一个函数，将一个字符串中的每个空格替换成“%20”。*/
    public class SolutionJZ2{
        /*
        public String replaceSpace(StringBuffer str) {
            return str.toString().replace(" ","%20");
        }*/
        public String replaceSpace(StringBuffer str){
            StringBuffer ret=new StringBuffer();
            for(int i=0;i<str.length();i++){
                char c=str.charAt(i);
                if(c==' ')
                    ret.append("%20");
                else
                    ret.append(c);

            }
            return ret.toString();
        }
    }
    /**在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
     * 每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，
     * 判断数组中是否含有该整数。*/
    public class SolutionJZ1{
        public boolean Find(int target, int [][] array) {
            int rows=array.length;
            int cols=array[0].length;
            if(rows==0||cols==0)
                return false;
            int row=0;
            int col=cols-1;
            while(row<rows&&col>=0){
                if(target==array[row][col])
                    return true;
                else if(target>array[row][col]){
                    row++;
                }else{
                    col--;
                }
            }
            return false;
        }
    }
}
