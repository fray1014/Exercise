import org.junit.Test;
import util.ListNode;

import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.logging.Handler;

public class JZOffer {
    @Test
    public void test(){
        ListNode l0=new ListNode(1);
        ListNode l1=new ListNode(2);
        ListNode l2=new ListNode(3);
        ListNode l3=new ListNode(3);
        ListNode l4=new ListNode(4);
        l0.next=l1;
        l1.next=l2;
        l2.next=l3;
        l3.next=l4;
        Solution s =new Solution();
        System.out.println(s.deleteDuplication(l0));
        HashMap<Integer,Integer> hm=new HashMap<>();
        Lock l = new ReentrantLock();
    }
    public class Solution{
        public ListNode deleteDuplication(ListNode pHead){
            ListNode res = new ListNode(0);
            res.next = pHead;
            ListNode cur = pHead.next;
            ListNode pre = pHead;
            while(cur!=null){
                if(cur.next!=null&&cur.val==cur.next.val){
                    while(cur.val==cur.next.val){
                        cur=cur.next;
                    }
                    cur=cur.next;
                    pre.next=cur;
                }else{
                    pre=cur;
                    cur=cur.next;
                }

            }
            return res.next;
        }
    }
}
