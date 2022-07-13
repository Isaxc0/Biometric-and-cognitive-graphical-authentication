import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.io.*;
import java.util.*;
import java.util.List;
import javax.swing.*;


/**
 * Pattern drawing component to be embedded in the main window
 *
 * @author Isaac Baldwin
 */
public class PatternDrawingComponent extends JPanel {
    private final boolean[][] grid = new boolean[3][3]; //boolean indication of whether a node is connected to enable colour changes
    private final Integer[][] systemPattern; //stage pattern
    private final Integer[][] userPattern = new Integer[3][3];  //participant drawn pattern
    private final Rectangle[][] hitbox = new Rectangle[3][3]; //rectangle around nodes for mouse detection
    private final List<List> allData; //all drawing data collected
    private List<List> currentData; //current drawing data for current drawing
    private final List<Line2D.Double> lines = new ArrayList<>(); //lines connected nodes
    //x and y values of previous and just connected nodes
    private Integer thisNodeX = 0;
    private Integer thisNodeY = 0;
    private Integer prevNodeX = 0;
    private Integer prevNodeY = 0;
    private Integer counter = 0; //number of correctly drawn patterns
    private final Thread thread2;
    private Integer nodeNum = 0; //number of nodes connected
    private boolean drawing = false;
    private boolean correct = false;
    private long startTime; //start time from when drawing began
    private final String ID; //participant ID
    private final Integer stage; //stage number


    /**
     * Instantiates a new Pattern drawing component.
     *
     * @param pattern stage pattern to draw
     * @param ID      participant id
     * @param stage   stage number
     * @param frame   JFrame of the GUI
     */
    public PatternDrawingComponent(Integer[][] pattern, String ID, Integer stage, JFrame frame) {
        systemPattern = pattern;
        this.ID = ID;
        this.stage = stage;
        allData = new ArrayList<>();
        currentData = new ArrayList<>();
        //allows constant update of graphics on separate thread
        Thread thread1 = new Thread(() -> { //allows constant update of graphics on separate thread
            try {
                while (true) {
                    repaint();
                }
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        });

        thread2 = new Thread() {
            public void run() { //records data on a separate thread
                try {
                    if (drawing) {
                        double mouseX = MouseInfo.getPointerInfo().getLocation().x - frame.getLocationOnScreen().x;
                        double mouseY = MouseInfo.getPointerInfo().getLocation().y - frame.getLocationOnScreen().y;
                        long stopTime = System.currentTimeMillis();
                        // add to the drawing data
                        currentData.add(Arrays.asList((int) mouseX, (int) mouseY, stopTime - startTime, 0));
                    }
                } catch (Exception e) {
                    System.out.println(e);
                }
            }
        };
        clearGrid();
        MouseMotionListener motionListener = new MouseAdapter() {
            @Override
            public void mouseDragged(MouseEvent m) {
                thread2.run();
                thisNodeX = m.getX();
                thisNodeY = m.getY();
                for (int i = 0; i < 3; ++i) {
                    for (int x = 0; x < 3; ++x) {
                        Rectangle node = hitbox[i][x];
                        if (userPattern[i][x] < 1) {
                            if (node.contains(m.getPoint())) {  // if mouse is in a node
                                nodeNum++;
                                drawing = true;
                                grid[i][x] = true;
                                userPattern[i][x] = nodeNum;
                                thisNodeX = (int) node.getCenterX();
                                thisNodeY = (int) node.getCenterY();

                                //move mouse position to center of node that has been connected
                                Robot r = null;
                                try {
                                    r = new Robot();
                                } catch (AWTException e) {
                                    e.printStackTrace();
                                }
                                r.mouseMove(thisNodeX + 7 + frame.getLocationOnScreen().x, thisNodeY + 80 + frame.getLocationOnScreen().y);

                                if (prevNodeX == 0) {
                                    prevNodeX = thisNodeX;
                                    prevNodeY = thisNodeY;
                                }
                                // add connection line between the previous connected node and most recent connected node
                                lines.add(new Line2D.Double(thisNodeX, thisNodeY, prevNodeX, prevNodeY));
                                prevNodeX = thisNodeX;
                                prevNodeY = thisNodeY;
                                long stopTime = System.currentTimeMillis();
                                // add to the drawing data
                                currentData.add(Arrays.asList(thisNodeX + 7, thisNodeY + 80, (stopTime - startTime), "1"));
                            }
                        }
                    }
                }
            }
        };
        MouseListener mouseListener = new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent m) {  // begin drawing
                currentData = new ArrayList<>(); // set current recorded data to empty
                startTime = System.currentTimeMillis();  // start time of drawing
            }

            @Override
            public void mouseReleased(MouseEvent m) {  // complete drawing
                correct = checkPattern();  // check whether pattern draw matched stage pattern
                if (correct) {
                    currentData.add(Arrays.asList("end", "end", "end", "end")); // add indication of drawing complete at end of data
                    allData.addAll(currentData);  // add all current drawing data to overall data for the stage
                }
                clearGrid();
                drawing = false;
                prevNodeX = 0;
                prevNodeY = 0;
            }
        };

        addMouseMotionListener(motionListener);
        addMouseListener(mouseListener);
        thread1.start();
        thread2.start();
    }


    /**
     * Save drawing data to file.
     *
     * @throws IOException the io exception
     */
    public void saveToFile() throws IOException {
        FileWriter writer = new FileWriter(ID + "_" + stage + "_data.txt");
        for (List data : allData) {
            String line = data.get(0) + " " + data.get(1) + " " + data.get(2) + " " + data.get(3) + "\n";
            writer.write(line);
        }
        writer.close();
    }

    /**
     * Paints graphics of lines and nodes
     */
    public void paint(Graphics graph) {
        super.paint(graph);
        Graphics2D g = (Graphics2D) graph;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setStroke(new BasicStroke(20, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                g.setColor(Color.decode("#373ade"));
                // create invisible squares around nodes to detect whether the mouse has entered a node
                hitbox[i][l] = new Rectangle();
                hitbox[i][l].setLocation(170 * (l + 1) + (20 * (l)), 170 * (i + 1) + (20 * (i)));
                hitbox[i][l].setSize(60, 60);

                if (!grid[i][l]) {  // change colour if node connected
                    g.setColor(Color.decode("#7EB5E6"));
                }
                g.setStroke(new BasicStroke(8));
                g.drawOval(170 * (l + 1) + (20 * l), 170 * (i + 1) + (20 * i), 60, 60);
                g.setColor(Color.decode("#373ade"));
                g.fillOval(190 * (l + 1), 190 * (i + 1), 20, 20);
            }
        }

        g.setStroke(new BasicStroke(15, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        if (drawing) { //draws a line from the last node pressed and the current mouse position
            g.drawLine(thisNodeX, thisNodeY, prevNodeX, prevNodeY);
        }
        for (Line2D.Double line : lines) { //draws line connecting drawn nodes
            g.draw(line);
        }
    }

    /**
     * Compares the participant drawn pattern and the system pattern
     *
     * @return return true if the pattern drawn by the user is valid
     */
    public boolean checkPattern() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (!userPattern[i][j].equals(systemPattern[i][j])) {
                    return false;
                }
            }
        }
        counter++;  // increment number of patterns drawn correctly
        return true;
    }


    /**
     * Is correct boolean.
     *
     * @return the boolean
     */
    public boolean isCorrect() {
        return correct;
    }

    /**
     * Gets counter (number of patterns drawn correctly).
     *
     * @return the counter
     */
    public Integer getCounter() {
        return counter;
    }

    /**
     * Clears the grid of lines and connected nodes
     */
    private void clearGrid() {
        for (int i = 0; i < 3; ++i) {
            for (int x = 0; x < 3; ++x) {
                grid[i][x] = false;
                userPattern[i][x] = 0;
            }
        }
        nodeNum = 0;
        lines.clear();
    }

}
