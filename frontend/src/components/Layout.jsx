import React, { useState } from 'react'
import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const navItems = [
  { to: '/', label: 'Dashboard', icon: '⬡' },
  { to: '/upload', label: 'Upload', icon: '↑' },
  { to: '/image-verify', label: 'Image Verify', icon: '🔍' },
  { to: '/claims', label: 'Claims', icon: '✓' },
  { to: '/tools', label: 'Tools', icon: '⚙' },
  { to: '/history', label: 'History', icon: '≡' },
  { to: '/compare', label: 'Compare', icon: '⇌' },
  { to: '/analytics', label: 'Analytics', icon: '◈' },
]

export default function Layout() {
  const { user, logoutUser } = useAuth()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)

  const handleLogout = () => {
    logoutUser()
    navigate('/login')
  }

  return (
    <div className="flex min-h-screen bg-slate-100">
      {/* Sidebar */}
      <aside className="hidden md:flex flex-col w-56 bg-white border-r border-slate-200 p-4 gap-1 shadow-sm">
        <div className="mb-6 px-2">
          <span className="text-lg font-bold tracking-tight text-indigo-600">FakeGuard</span>
          <div className="text-xs text-slate-400 mt-0.5">News Detection Platform</div>
        </div>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition ${
                isActive
                  ? 'bg-indigo-50 text-indigo-700 border border-indigo-100'
                  : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800'
              }`
            }
          >
            <span className="text-base leading-none">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
        <div className="mt-auto pt-4 border-t border-slate-100">
          <p className="text-xs text-slate-400 mb-2 px-1 truncate">{user?.email}</p>
          <button
            onClick={handleLogout}
            className="w-full text-sm px-3 py-2 rounded-lg text-slate-500 hover:bg-red-50 hover:text-red-600 transition text-left"
          >
            Sign out
          </button>
        </div>
      </aside>

      {/* Mobile top bar */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-50 bg-white border-b border-slate-200 flex items-center justify-between px-4 py-3 shadow-sm">
        <span className="text-indigo-600 font-bold text-sm tracking-tight">FakeGuard</span>
        <button onClick={() => setOpen(!open)} className="text-slate-500 text-lg font-bold">≡</button>
      </div>

      {open && (
        <div className="md:hidden fixed top-12 left-0 right-0 z-40 bg-white border-b border-slate-200 flex flex-col p-3 gap-1 shadow-md">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                `flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium ${
                  isActive ? 'bg-indigo-50 text-indigo-700' : 'text-slate-500'
                }`
              }
            >
              <span>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
          <button onClick={handleLogout} className="px-3 py-2 rounded-lg text-sm text-red-500 text-left">
            Sign out
          </button>
        </div>
      )}

      <main className="flex-1 p-6 mt-12 md:mt-0 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}
