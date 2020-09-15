import java.util.*;
import util.*;
import org.junit.Test;
public class BinaryTree {
    @Test
    public void test(){
        TreeNode t0=new TreeNode(0);
        TreeNode t1=new TreeNode(1);
        TreeNode t2=new TreeNode(2);
        TreeNode t3=new TreeNode(3);
        TreeNode t4=new TreeNode(4);
        TreeNode t5=new TreeNode(5);
        TreeNode t6=new TreeNode(6);
        TreeNode t7=new TreeNode(7);
        t0.left=t1;
        t0.right=t2;
        t1.left=t3;
        t1.right=t4;
        t2.left=t5;
        t2.right=t6;
        t4.left=t7;
        levelOrderTraverse(t0);
    }
    /**递归前序*/
    public void preOrderTraverse1(TreeNode root){
        if(root!=null){
            System.out.println(root.val);
            preOrderTraverse1(root.left);
            preOrderTraverse1(root.right);
        }
    }
    /**非递归前序*/
    public void preOrderTraverse2(TreeNode root){
        Stack<TreeNode> stack=new Stack<>();
        while(root!=null||!stack.isEmpty()){
            if(root!=null){
                System.out.println(root.val);
                stack.push(root);
                root=root.left;
            }else{
                root=stack.pop().right;
            }
        }
    }
    /**递归中序*/
    public void inOrderTraverse1(TreeNode root){
        if(root!=null){
            inOrderTraverse1(root.left);
            System.out.println(root.val);
            inOrderTraverse1(root.right);
        }
    }
    /**非递归中序*/
    public void inOrderTraverse2(TreeNode root){
        Stack<TreeNode> stack=new Stack<>();
        while(root!=null||!stack.isEmpty()){
            if(root!=null){
                stack.push(root);
                root=root.left;
            }else{
                System.out.println(stack.peek().val);
                root=stack.pop().right;
            }
        }
    }
    /**递归后序*/
    public void postOrderTraverse1(TreeNode root){
        if(root!=null){
            postOrderTraverse1(root.left);
            postOrderTraverse1(root.right);
            System.out.println(root.val);
        }
    }
    /**非递归后序*/
    public void postOrderTraverse2(TreeNode root){
        TreeNode cur, pre = null;

        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);

        while (!stack.empty()) {
            cur = stack.peek();
            if ((cur.left == null && cur.right == null) || (pre != null && (pre == cur.left || pre == cur.right))) {
                System.out.println(cur.val);
                stack.pop();
                pre = cur;
            } else {
                if (cur.right != null)
                    stack.push(cur.right);
                if (cur.left != null)
                    stack.push(cur.left);
            }
        }
    }
    /**层次遍历*/
    public void levelOrderTraverse(TreeNode root){
        Queue<TreeNode> queue=new LinkedList<>();
        queue.offer(root);
        while(!queue.isEmpty()){
            TreeNode cur=queue.poll();
            System.out.println(cur.val);
            if(cur.left!=null)
                queue.offer(cur.left);
            if(cur.right!=null)
                queue.offer(cur.right);
        }
    }
}
