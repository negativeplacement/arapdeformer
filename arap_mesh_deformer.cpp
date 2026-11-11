#include <iostream>
#include <queue>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <igl/cotmatrix.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/project.h>
#include <igl/read_triangle_mesh.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <imgui.h>

int main(int argc, char* argv[]){
    Eigen::MatrixXi face, original_face;
    Eigen::MatrixXd vertex, original_vertex;
    if(!igl::read_triangle_mesh(argv[1], vertex, face)){
        std::cerr << "Error Loading Mesh" << std::endl;
        return 0;
    }
    original_face = face;
    original_vertex = vertex;
    Eigen::MatrixXd deformed = vertex;
    Eigen::SparseMatrix<double> cotangent_weight;
    igl::cotmatrix(vertex, face, cotangent_weight);
    cotangent_weight = -cotangent_weight;
    std::vector<std::vector<int>> adjacency_list(vertex.rows());
    std::vector<std::vector<double>> adjacency_weight(vertex.rows());
    for(int _ = 0; _ < cotangent_weight.outerSize(); ++_){
        for(Eigen::SparseMatrix<double>::InnerIterator iterator(cotangent_weight, _); iterator; ++iterator){
            if(_ != iterator.index()){
                adjacency_list[_].push_back(iterator.index());
                adjacency_weight[_].push_back(iterator.value());
            }
        }
    }
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> coefficient;
    bool constraint = 0, target = 0, mapping = 0, modified = 0, highlight = 0, deform = 0;
    double penalty = 1e6;
    std::string constraint_label, target_label;
    int start_vertex = -1;
    Eigen::RowVector3d target_projection;
    std::set<int> constraint_vertex, target_vertex, selected_vertex;
    std::map<int, Eigen::RowVector3d> target_map, selected_projection;
    std::vector<Eigen::Matrix3d> rotation(vertex.rows());
    auto optimization = [&](){
        Eigen::SparseMatrix<double> arbitrary = cotangent_weight;
        Eigen::VectorXd D = Eigen::VectorXd::Zero(vertex.rows());
        for(int _ : constraint_vertex) D(_) = penalty;
        for(int _ : target_vertex) D(_) = penalty;
        arbitrary += D.asDiagonal();
        coefficient.compute(arbitrary);
        return arbitrary;
    };
    enum class selection {none, constraint, target};
    selection procedure = selection::none;
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);
    viewer.data().set_mesh(vertex, face);
    viewer.data().set_colors(Eigen::RowVector3d(0.7, 0.7, 0.7));
    viewer.data().line_color = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 1.0f);
    igl::opengl::glfw::imgui::ImGuiMenu imgui_menu;
    imgui_plugin.widgets.push_back(&imgui_menu);
    Eigen::Vector2f select_start, select_end;
    imgui_menu.callback_draw_viewer_menu = [&](){
        auto& core = viewer.core();
        float display_height = 80.0f;
        ImGui::SetNextWindowPos(ImVec2(0, viewer.core().viewport(3) - display_height), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(viewer.core().viewport(2), display_height), ImGuiCond_Always);
        ImGui::Begin("Deformer", nullptr, ImGuiWindowFlags_NoTitleBar);
        float space = ImGui::GetStyle().ItemSpacing.x;
        float button_width = (ImGui::GetContentRegionAvail().x - (2 * space)) / 3.0f;
        if(mapping && procedure == selection::constraint) constraint_label = "Mapping...";
        else constraint_label = "Select Constraint";
        if(ImGui::Button(constraint_label.c_str(), ImVec2(button_width/2, 0))){
            mapping = 1;
            procedure = selection::constraint;
        }
        ImGui::SameLine(0, 5.0f);
        if(ImGui::Button("Clear Constraint", ImVec2(button_width/2, 0))){
            constraint = 0;
            constraint_vertex.clear();
            procedure = selection::none;
        }
        ImGui::Separator();
        if(mapping && procedure == selection::target) target_label = "Mapping...";
        else target_label = "Select Target";
        if(ImGui::Button(target_label.c_str(), ImVec2(button_width/2, 0))){
            mapping = 1;
            procedure = selection::target;
        }
        ImGui::SameLine(0, 5.0f);
        if(ImGui::Button("Clear Target", ImVec2(button_width/2, 0))){
            target = 0;
            target_map.clear();
            target_vertex.clear();
            procedure = selection::none;
        }
        ImGui::SameLine(0, 5.0f);
        if(modified) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
        if(ImGui::Button("Reset Mesh", ImVec2(button_width/2, 0))){
            if(modified){
                constraint_vertex.clear();
                target_map.clear();
                target_vertex.clear();
                deformed = vertex;
                viewer.data().set_vertices(deformed);
                viewer.data().set_points(Eigen::MatrixXd(0,3), Eigen::MatrixXd(0,3));
                ImGui::PopStyleColor();
            }
        } else if(modified){
            ImGui::PopStyleColor();
        }
        ImGui::SameLine(0, 5.0f);
        if(ImGui::Button("Save Mesh", ImVec2(button_width/2, 0))){
            if(!igl::write_triangle_mesh("Deformed Mesh.obj", deformed, face)){
                std::cerr << "Error Saving Mesh" << std::endl;
            }
        }
        if(highlight){
            select_end = Eigen::Vector2f(ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y);
            ImGui::GetForegroundDrawList()->AddRectFilled(ImVec2(select_start.x(), select_start.y()), ImVec2(select_end.x(), select_end.y()), IM_COL32(0, 130, 255, 50));
            ImGui::GetForegroundDrawList()->AddRect(ImVec2(select_start.x(), select_start.y()), ImVec2(select_end.x(), select_end.y()), IM_COL32(0, 130, 255, 255));
        }
        ImGui::End();
    };
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& view, int button, int modifier) -> bool {
        if(ImGui::GetIO().WantCaptureMouse) return 1;
        if(procedure != selection::none && button == GLFW_MOUSE_BUTTON_LEFT){
            std::cout << "Entering Mouse highlight" << std::endl;
            highlight = 1;
            select_start = Eigen::Vector2f(ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y);
            select_end = select_start;
            std::cout << "Exiting Mouse highlight" << std::endl;
            return 1;
        } else if(procedure == selection::none && button == GLFW_MOUSE_BUTTON_LEFT){
            if(constraint_vertex.empty() || target_vertex.empty()) return 0;
            std::cout << "Entering Mouse Down for stretch" << std::endl;
            int face_, reference;
            Eigen::Vector3f baycentric, vector_projection;
            if(igl::unproject_onto_mesh(Eigen::Vector2f(view.current_mouse_x, view.core().viewport(3) - view.current_mouse_y), view.core().view, view.core().proj, view.core().viewport, vertex.cast<float>(), face, face_, baycentric)){
                baycentric.maxCoeff(&reference);
                int vertex_ = face(face_, reference);
                if(target_vertex.count(vertex_)){
                    deform = 1;
                    start_vertex = vertex_;
                    Eigen::Vector3f vector_plane = igl::project(Eigen::Vector3f(vertex.row(vertex_).cast<float>().transpose()), view.core().view, view.core().proj, view.core().viewport);
                    igl::unproject(Eigen::Vector3f(view.current_mouse_x, view.core().viewport(3) - view.current_mouse_y, vector_plane(2)), view.core().view, view.core().proj, view.core().viewport, vector_projection);
                    target_projection = vector_projection.cast<double>().transpose();
                    selected_vertex.clear();
                    selected_projection.clear();
                    std::queue<int> queue_;
                    queue_.push(start_vertex);
                    std::set<int> visited;
                    visited.insert(vertex_);
                    std::cout << "BFS" << std::endl;
                    while(!queue_.empty()){
                        vertex_ = queue_.front();
                        queue_.pop();
                        selected_vertex.insert(vertex_);
                        selected_projection[vertex_] = target_map[vertex_];
                        for(int _ : adjacency_list[vertex_]){
                            if(target_vertex.count(_) && visited.find(_) == visited.end()){
                                visited.insert(_);
                                queue_.push(_);
                            }
                        }
                    }
                    std::cout << "BFS: V=" << selected_vertex.size() << " P= " << selected_projection.size() << std::endl;
                    std::cout << "Exiting Mouse Down For Stretch" << std::endl;
                    return 1;
                }
            }
        }
        return 0;
    };
    viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer& view, int button, int modifier) -> bool {
        if(constraint_vertex.empty() || target_vertex.empty()) return 0;
        if(deform && start_vertex != -1){
            std::cout << "Entering Mouse Move" << std::endl;
            Eigen::Vector3f vector_plane = igl::project(Eigen::Vector3f(vertex.row(start_vertex).cast<float>().transpose()), view.core().view, view.core().proj, view.core().viewport);
            Eigen::Vector3f vector_projection;
            igl::unproject(Eigen::Vector3f(view.current_mouse_x, view.core().viewport(3) - view.current_mouse_y, vector_plane(2)), view.core().view, view.core().proj, view.core().viewport, vector_projection);
            Eigen::RowVector3d delta = vector_projection.cast<double>().transpose() - target_projection;
            std::cout << "Delta: " << delta.x() << ", " << delta.y() << ", " << delta.z() << " | Norm: " << delta.norm() << std::endl;
            for(int _ : selected_vertex) target_map[_] = selected_projection[_] + delta;
            for(int iteration = 0; iteration < 10; ++iteration){
                #pragma omp parallel for
                for(int _ = 0; _ < vertex.rows(); _++){
                    Eigen::Matrix3d computation = Eigen::Matrix3d::Zero();
                    for(std::size_t k = 0; k < adjacency_list[_].size(); ++k) computation += adjacency_weight[_][k] * ((vertex.row(_) - vertex.row(adjacency_list[_][k])).transpose()  * (deformed.row(_) - deformed.row(adjacency_list[_][k])));
                    Eigen::JacobiSVD<Eigen::Matrix3d> singular_value_decomposition(computation, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    rotation[_] = singular_value_decomposition.matrixV() * singular_value_decomposition.matrixU().transpose();
                    if(rotation[_].determinant() < 0){
                        Eigen::Matrix3d U = singular_value_decomposition.matrixU();
                        U.col(2) *= -1.0;
                        rotation[_] = singular_value_decomposition.matrixV() * U.transpose();
                    }
                }
                Eigen::MatrixXd solution = Eigen::MatrixXd::Zero(vertex.rows(), 3);
                for(int _ = 0; _ < vertex.rows(); _++){
                    Eigen::Vector3d vector = Eigen::Vector3d::Zero();
                    for(std::size_t k = 0; k < adjacency_list[_].size(); ++k) vector += (adjacency_weight[_][k] / 2.0) * (rotation[_] + rotation[adjacency_list[_][k]]) * (vertex.row(_) - vertex.row(adjacency_list[_][k])).transpose();
                    solution.row(_) = vector;
                }
                for(int _ : constraint_vertex) solution.row(_) += penalty * vertex.row(_);
                for(auto& [_, pos] : target_map) solution.row(_) += penalty * pos;
                deformed = coefficient.solve(solution);
            }
            view.data().set_vertices(deformed);
            view.data().point_size = 10.0f;
            Eigen::MatrixXd vertex_point(constraint_vertex.size() + target_vertex.size(), 3), colour_point(vertex_point.rows(), 3);
            int x = 0;
            for(int _ : constraint_vertex){
                vertex_point.row(x) = deformed.row(_);
                colour_point.row(x) = Eigen::RowVector3d(1,0,0);
                x++;
            }
            for(int _ : target_vertex){
                vertex_point.row(x) = target_map[_];
                colour_point.row(x) = Eigen::RowVector3d(0,1,0);
                x++;
            }
            view.data().set_points(vertex_point, colour_point);
            modified = 1;
            std::cout << "Exiting Mouse Move" << std::endl;
            return 1;
        }
        return 0;
    };
    viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer& view, int button, int modifier) -> bool {
        if(ImGui::GetIO().WantCaptureMouse) return 1;
        if(highlight){
            std::cout << "Entering Mouse Up" << std::endl;
            float min_x = std::min(select_start.x(), select_end.x());
            float max_x = std::max(select_start.x(), select_end.x());
            float min_y = std::min(select_start.y(), select_end.y());
            float max_y = std::max(select_start.y(), select_end.y());
            for(int _ = 0; _ < vertex.rows(); _++){
                Eigen::Vector3f vector_plane = igl::project(Eigen::Vector3f(vertex.row(_).cast<float>()), view.core().view, view.core().proj, view.core().viewport);
                float py = view.core().viewport(3) - vector_plane.y(); 
                if(vector_plane.x() >= min_x && vector_plane.x() <= max_x && py >= min_y && py <= max_y){
                    if(procedure == selection::constraint){
                        constraint_vertex.insert(_);
                    } else if(procedure == selection::target){
                        Eigen::Vector3f vector_projection;
                        igl::unproject(Eigen::Vector3f(view.current_mouse_x, view.core().viewport(3) - view.current_mouse_y, vector_plane(2)), view.core().view, view.core().proj, view.core().viewport, vector_projection);
                        //target_map[_] = vector_projection.cast<double>().transpose();
                        target_map[_] = deformed.row(_);
                        target_vertex.insert(_);
                    }
                    std::cout << "Vertex " << _ << " hit! Screen: " << vector_plane.x() << "," << py << std::endl;
                }
            }
            optimization();
            Eigen::MatrixXd vertex_point(constraint_vertex.size() + target_vertex.size(), 3), colour_point(vertex_point.rows(), 3);
            int x = 0;
            if(!constraint_vertex.empty()) constraint = 1;
            for(int _ : constraint_vertex){
                vertex_point.row(x) = deformed.row(_);
                colour_point.row(x) = Eigen::RowVector3d(1,0,0);
                x++;
            }
            if(!target_vertex.empty()) target = 1;
            for(int _ : target_vertex){
                vertex_point.row(x) = deformed.row(_);
                colour_point.row(x) = Eigen::RowVector3d(0,1,0);
                x++;
            }
            view.data().set_points(vertex_point, colour_point);
            mapping = 0;
            highlight = 0;
            procedure = selection::none;
            std::cout << "Exiting Mouse Up" << std::endl;
            return 1;
        }
        if(deform){
            deform = 0;
            start_vertex = -1;
            selected_vertex.clear();
            return 1;
        }
        return 0;
    };
    viewer.launch_init(0, "ARAP Mesh Deformer", 800, 600);
    //viewer.core().viewport = Eigen::Vector4f(0, 0, 800, 600);
    viewer.launch_rendering(1);
    viewer.launch_shut();
    return 0;
}
//c_cpp -file arap_mesh_deformer.cpp,glad.c,imgui.cpp,imgui_draw.cpp,imgui_tables.cpp,imgui_widgets.cpp,imgui_impl_glfw.cpp,imgui_impl_opengl3.cpp -link libigl -flag "-fopenmp","-O2","-Wa,-mbig-obj","-Wl,--stack,16777216" -args cylinder.obj