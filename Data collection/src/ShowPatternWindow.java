import javax.swing.*;
import java.awt.*;
import java.awt.geom.Line2D;
import java.util.ArrayList;
import java.util.List;

/**
 * Displays the pattern for the current stage
 *
 * @author Isaac Baldwin
 */
public class ShowPatternWindow extends JPanel {
    private final Integer[][] pattern; //stage pattern
    private final Rectangle[][] rectangles = new Rectangle[3][3]; //rectangle around nodes for mouse detection
    private final List<Line2D.Double> lines = new ArrayList<>();  //lines connected nodes

    /**
     * Instantiates a new Show pattern window.
     *
     * @param stagePattern the generated pattern
     */
    public ShowPatternWindow(Integer[][] stagePattern) {
        pattern = stagePattern;
        generateDisplay();
    }

    /**
     * Generates the graphical components to display the stage pattern
     */
    private void generateDisplay() {
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                rectangles[i][l] = new Rectangle();
            }
        }
        Integer counter = 1;
        Integer maxValue = getPatternLength();
        Integer thisX, thisY;
        Integer prevX = 0, prevY = 0;

        // get lines to connect between the nodes for the stage pattern
        while (counter <= maxValue) {
            for (int i = 0; i < 3; ++i) {
                for (int l = 0; l < 3; ++l) {
                    rectangles[i][l] = new Rectangle();
                    rectangles[i][l].setLocation(170 * (l + 1) + (20 * (l)), 170 * (i + 1) + (20 * (i)));
                    rectangles[i][l].setSize(60, 60);
                    if (pattern[i][l].equals(counter)) {
                        counter++;
                        thisX = (int) rectangles[i][l].getCenterX();
                        thisY = (int) rectangles[i][l].getCenterY();
                        if (pattern[i][l] == 1) {  // no line needs to be drawn for the first node
                            prevX = thisX;
                            prevY = thisY;
                        }
                        lines.add(new Line2D.Double(thisX, thisY, prevX, prevY));
                        prevX = thisX;
                        prevY = thisY;
                    }
                }
            }
        }
    }

    private Integer getPatternLength() {
        int max = 0;
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                if (pattern[i][l] > max) {
                    max = pattern[i][l];
                }
            }
        }
        return max;
    }

    public void paint(Graphics graph) {
        Graphics2D g = (Graphics2D) graph;
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setStroke(new BasicStroke(20, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        for (int i = 0; i < 3; ++i) {
            for (int l = 0; l < 3; ++l) {
                g.setColor(Color.decode("#373ade"));
                rectangles[i][l] = new Rectangle();
                rectangles[i][l].setLocation(170 * (l + 1) + (20 * (l)), 170 * (i + 1) + (20 * (i)));
                rectangles[i][l].setSize(60, 60);

                if (pattern[i][l] == 0) {
                    g.setColor(Color.decode("#7EB5E6"));
                }
                if (pattern[i][l] == 1) {
                    g.setColor(Color.decode("#228b22"));
                }
                if (pattern[i][l].equals(getPatternLength())) {
                    g.setColor(Color.decode("#ff0000"));
                }

                g.setStroke(new BasicStroke(8));
                g.drawOval(170 * (l + 1) + (20 * l), 170 * (i + 1) + (20 * i), 60, 60);
                g.setColor(Color.decode("#373ade"));
                g.fillOval(190 * (l + 1), 190 * (i + 1), 20, 20);
            }
        }

        g.setStroke(new BasicStroke(15, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        for (Line2D.Double line : lines) { //draws line connecting user selected nodes
            g.draw(line);
        }
    }
}
